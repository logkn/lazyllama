import socket

import psutil
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)

from src.core.models.alias import Alias, Backend
from src.core.servers.base_server import BaseServer, ServerStatus
import json
from src.constants import RESOURCE_CACHE_PATH

DEFAULT_PORTS = {
    Backend.OLLAMA: 11434,
    Backend.LLAMACPP: 8000,
    # Add more as needed
}


class ServerManager:
    def __init__(self):
        self.running_servers: list[BaseServer] = []
        # Cache measured resource models per (backend, model_name, command_params)
        # command_params is converted to tuple for hashing
        self.resource_models: dict[
            tuple[Backend, str, tuple[str, ...]], tuple[float, float, float, float]
        ] = {}
        # Load persistent resource model cache
        self._load_resource_cache()

        self.total_ram_mb: float = get_total_ram_mb()
        self.total_vram_mb: float = get_total_vram_mb()

    async def get_or_start_server(self, alias: Alias) -> BaseServer:
        for server in self.running_servers:
            if await server.check_running() and self.is_compatible(alias, server.alias):
                return server

        r0, r1, v0, v1 = self.get_or_measure_resource_model(alias)
        R_need = r0 + r1 * alias.n_ctx
        V_need = v0 + v1 * alias.n_ctx

        R_used, V_used = await self.get_current_usage()
        R_free = self.total_ram_mb - R_used
        V_free = self.total_vram_mb - V_used

        print("Post-eviction:")
        print(f"Available: R={R_free}, V={V_free}")

        if R_free < R_need or V_free < V_need:
            # 🔍 Debug: print all currently tracked servers
            print("RUNNING SERVERS:")
            for s in self.running_servers:
                print(
                    f"  {s.alias.name} | status: {s.status} | running: {await s.check_running()}"
                )
            evicted = await self.evict_servers(R_need - R_free, V_need - V_free)
            for evicted_server in evicted:
                for running_server in self.running_servers:
                    if running_server.alias == evicted_server.alias:
                        print("Evicting:", evicted_server.alias.name)
                        print("  is in list:", evicted_server in self.running_servers)
                        print(
                            "  running_servers:", [id(s) for s in self.running_servers]
                        )
                        print("  target server id:", id(evicted_server))
                        self.running_servers.remove(running_server)
                        break
                await evicted_server.stop_and_wait()
            # Recheck after eviction
            R_used, V_used = await self.get_current_usage()
            R_free = self.total_ram_mb - R_used
            V_free = self.total_vram_mb - V_used

        if R_free < R_need or V_free < V_need:
            print("Eviction failed:")
            print(f"Needed: R={R_need}, V={V_need}")
            print(f"Available: R={R_free}, V={V_free}")
            print(f"Servers: {[s.alias.name for s in self.running_servers]}")
            raise RuntimeError(
                "No combination of servers can satisfy resource requirements"
            )

        port = self.get_free_port(alias.model.backend)
        server = self.create_server(alias, port)
        await server.start_and_wait()
        self.running_servers.append(server)
        return server

    def is_compatible(self, request: Alias, running: Alias) -> bool:
        return (
            request.model == running.model
            and request.command_params == running.command_params
            and request.n_ctx <= running.n_ctx
        )

    async def get_current_usage(self) -> tuple[float, float]:
        total_R = 0.0
        total_V = 0.0

        for server in self.running_servers:
            running = await server.check_running()
            print(f"[USAGE] {server.alias.name} | running={running}")
            if not running:
                continue

            r0, r1, v0, v1 = self.get_or_measure_resource_model(server.alias)
            x = server.alias.n_ctx
            total_R += r0 + r1 * x
            total_V += v0 + v1 * x

        return total_R, total_V

    def get_or_measure_resource_model(
        self, alias: Alias
    ) -> tuple[float, float, float, float]:
        # Key the cache by backend, model_name, and command_params
        key = (
            alias.model.backend,
            alias.model.model_name,
            tuple(alias.command_params),
        )
        if key in self.resource_models:
            return self.resource_models[key]
        model = self.measure_resource_model(alias)
        self.resource_models[key] = model
        # Persist updated resource model cache
        self._persist_resource_cache()
        return model

    def measure_resource_model(self, alias: Alias) -> tuple[float, float, float, float]:
        raise NotImplementedError("Resource model measurement not implemented")

    def _load_resource_cache(self) -> None:
        path = RESOURCE_CACHE_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                for key_str, vals in data.items():
                    parts = key_str.split("::")
                    if len(parts) != 3:
                        continue
                    backend_str, model_name, params_str = parts
                    command_params = params_str.split(",") if params_str else []
                    try:
                        backend = Backend(backend_str)
                    except ValueError:
                        continue
                    key = (backend, model_name, tuple(command_params))
                    if isinstance(vals, (list, tuple)) and len(vals) == 4:
                        self.resource_models[key] = tuple(vals)
        except Exception:
            pass

    def _persist_resource_cache(self) -> None:
        path = RESOURCE_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, list[float]] = {}
        for key, vals in self.resource_models.items():
            backend, model_name, command_params = key
            params_str = ",".join(command_params)
            key_str = f"{backend.value}::{model_name}::{params_str}"
            data[key_str] = list(vals)
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    async def evict_servers(
        self, R_deficit: float, V_deficit: float
    ) -> list[BaseServer]:
        candidates: list[tuple[float, BaseServer, float, float]] = []
        for server in self.running_servers:
            if server.status != ServerStatus.RUNNING:
                continue
            r0, r1, v0, v1 = self.get_or_measure_resource_model(server.alias)
            x = server.alias.n_ctx
            R_i = r0 + r1 * x
            V_i = v0 + v1 * x
            w_i = max(R_i / max(R_deficit, 1e-6), V_i / max(V_deficit, 1e-6))
            candidate = (w_i, server, R_i, V_i)
            candidates.append(candidate)

        print("Eviction candidates:")
        for w_i, server, R_i, V_i in candidates:
            print(f"  {server.alias.name} | R={R_i:.1f}, V={V_i:.1f}, w={w_i:.2f}")
        candidates.sort(reverse=True, key=lambda tup: tup[0])
        selected: list[BaseServer] = []
        R_total = 0.0
        V_total = 0.0
        for _, server, R_i, V_i in candidates:
            selected.append(server)
            R_total += R_i
            V_total += V_i
            if R_total >= R_deficit and V_total >= V_deficit:
                break
        print("Selected for eviction:")
        for server in selected:
            print(f"  {server.alias.name}")

        if not selected:
            raise RuntimeError(
                "No servers can be evicted to satisfy resource requirements"
            )
        return selected

    def get_free_port(self, backend: Backend, max_tries: int = 100) -> int:
        base_port = DEFAULT_PORTS[backend]
        for offset in range(max_tries):
            port = base_port + offset
            if not any(s.port == port for s in self.running_servers):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(("localhost", port))
                        return port
                    except OSError:
                        continue
        raise RuntimeError(f"No free ports available for backend {backend}.")

    def create_server(self, alias: Alias, port: int) -> BaseServer:
        raise NotImplementedError("Server creation not implemented yet")


def get_total_ram_mb() -> float:
    """Return total system RAM in MB."""
    return psutil.virtual_memory().total / 1024 / 1024


def get_total_vram_mb() -> float:
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # First GPU
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        return mem_info.total / 1024 / 1024
    except Exception:
        return 0.0
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass
