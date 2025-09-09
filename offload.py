from typing import Callable

import gc
from functools import cached_property
from tqdm.auto import tqdm
from pathlib import Path
import json

import torch
import torch.nn as nn

# --- Constants and Type Definitions ---
ORIG_FORWARD_ATTR = "tf_orig_forward"


# --- Helper Functions ---
def clear_model_patches(model: nn.Module, /) -> None:
    """Restores the original forward methods for all patched modules in a model."""

    for module in model.modules():
        if hasattr(module, ORIG_FORWARD_ATTR):
            module.forward = getattr(module, ORIG_FORWARD_ATTR)
            delattr(module, ORIG_FORWARD_ATTR)


def store_original_forward(module: nn.Module, /) -> Callable:
    """Saves the original forward method of a module before applying a patch."""

    if not hasattr(module, ORIG_FORWARD_ATTR):
        setattr(module, ORIG_FORWARD_ATTR, module.forward)
    return getattr(module, ORIG_FORWARD_ATTR)


def is_leaf(module: nn.Module, /) -> bool:
    has_children = len(list(module.children())) > 0
    has_parameters = len(list(module.parameters())) + len(list(module.buffers())) > 0

    return not has_children and has_parameters


class AutoCPUOffloadManager:
    def __init__(
        self,
        model: nn.Module,
        /,
        window_size: int = 2,
        device: str | torch.device = "cuda",
        pin_memory: bool = True,
    ) -> None:
        self.model = model
        self.window_size = window_size
        self.device = device = torch.device(device)
        self.cpu_device = torch.device("cpu")
        self.pin_memory = pin_memory

        # ?
        self.all_modules: set[nn.Module] = set(module for module in self.model.modules() if module is not model)
        self.all_non_leaf_modules: set[nn.Module] = set(module for module in self.all_modules if not is_leaf(module))

        self.leaf_modules_mapping: dict[str, nn.Module] = {name: module for (name, module) in self.model.named_modules() if is_leaf(module)}
        self.leaf_modules: set[nn.Module] = set(self.leaf_modules_mapping.values())
        self.leaf_module_names: set[str] = set(self.leaf_modules_mapping.keys())
        self.num_leaf_modules = num_leaf_modules = len(self.leaf_modules_mapping)
        assert len(self.leaf_module_names) == num_leaf_modules, "Duplicate module names found."

        # --- State Management ---
        self.is_tracing = True
        self.execution_order: list[str] = []

        # --- Hardware Resource Management ---
        self.compute_stream = torch.cuda.current_stream(self.device)
        self.memory_stream = torch.cuda.Stream(self.device)

        self.compute_events = [torch.cuda.Event(blocking=False) for _ in range(num_leaf_modules)]
        self.load_events = [torch.cuda.Event(blocking=False) for _ in range(num_leaf_modules)]

    def initialize(self) -> None:
        clear_model_patches(self.model)
        self.set_pin_memory()
        self.patch_model()
        self.patch_leaf_modules()

        self.load_execution_order()

        self.clear_cache()

    @cached_property
    def config_path(self) -> Path:
        try:
            folder = Path(__file__).parent / "offload_configs"
        except NameError:
            folder = Path(".") / "user-data"

        # Create a unique name based on the model's class
        full_model_name = f"{self.model.__class__.__module__}.{self.model.__class__.__qualname__}"

        return folder / f"{full_model_name}.n{self.num_leaf_modules}.json"

    def save_execution_order(self) -> None:
        path = self.config_path
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.execution_order, f, indent=4)

    def load_execution_order(self) -> None:
        if self.config_path.exists():
            execution_order: list[str] = json.load(self.config_path.open("r", encoding="utf-8"))

            if set(execution_order) != self.leaf_module_names:
                return

            part = self.config_path.stem.split(".")[-1]
            assert part.startswith("n")
            num_leaf_modules = int(part[1:])
            if int(num_leaf_modules) != self.num_leaf_modules:
                return

            self.execution_order = execution_order

            self.is_tracing = False
            self.initial_preload()

    def set_pin_memory(self) -> None:
        """Moves all offloadable modules to CPU and pins their memory."""

        for module in tqdm(self.leaf_modules, desc="Pinning Memory for Offloadable Modules"):
            module.to(self.cpu_device, non_blocking=True)

            if self.pin_memory:
                for param in module.parameters():
                    param.data = param.data.pin_memory()
                for buffer in module.buffers():
                    buffer.data = buffer.data.pin_memory()

    def clear_cache(self) -> None:
        torch.cuda.synchronize()

        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    def patch_model(self) -> None:
        """Wraps the model's top-level forward method to manage tracing and execution state."""

        original_model_forward = store_original_forward(self.model)

        def patched_forward(*args, **kwargs):
            if self.is_tracing:
                output = original_model_forward(*args, **kwargs)
                self.finalize_tracing()

                return output

            # --- Pipelined Execution Run ---
            with torch.cuda.stream(self.compute_stream):
                output = original_model_forward(*args, **kwargs)

            return output

        self.model.forward = patched_forward

    def finalize_tracing(self) -> None:
        """Saves the execution order and prepares for pipelined execution."""

        self.is_tracing = False

        self.initial_preload()
        self.save_execution_order()

    def initial_preload(self) -> None:
        """Pre-fetches the initial window of modules to the GPU."""

        num_to_preload = min(self.window_size, len(self.execution_order))
        for idx in range(num_to_preload):
            self.load_module(idx)

    def get_name_module(self, idx: int, /) -> tuple[str, nn.Module]:
        if idx < 0 or idx >= len(self.execution_order):
            raise IndexError(f"Index {idx} is out of range.")

        name = self.execution_order[idx]
        module = self.leaf_modules_mapping[name]

        return name, module

    def load_module(self, idx: int, /) -> None:
        name, module = self.get_name_module(idx)
        with torch.cuda.stream(self.memory_stream):
            module.to(self.device, non_blocking=True)
            self.load_events[idx].record(self.memory_stream)

    def unload_module(self, idx: int, /) -> None:
        name, module = self.get_name_module(idx)
        with torch.cuda.stream(self.memory_stream):
            self.memory_stream.wait_event(self.compute_events[idx])  # pyright: ignore[reportArgumentType]
            module.to(self.cpu_device, non_blocking=True)

    def patch_leaf_modules(self) -> None:
        for name in tqdm(self.leaf_module_names, desc="Patching Leaf Modules"):
            module = self.leaf_modules_mapping[name]
            original_forward = store_original_forward(module)

            def make_patched_forward(name: str, module: nn.Module, original_forward: Callable):
                def patched_forward(*args, **kwargs):
                    if self.is_tracing:
                        return self.trace_forward(name, module, original_forward, *args, **kwargs)
                    return self.managed_forward(name, module, original_forward, *args, **kwargs)

                return patched_forward

            module.forward = make_patched_forward(name, module, original_forward)

    def trace_forward(self, name: str, module: nn.Module, original_forward: Callable, *args, **kwargs):
        """Logic for the tracing phase: record execution and run synchronously."""

        self.execution_order.append(name)

        module.to(self.device)
        output = original_forward(*args, **kwargs)
        module.to(self.cpu_device)

        return output

    def managed_forward(self, name: str, module: nn.Module, original_forward: Callable, *args, **kwargs):
        cidx = self.execution_order.index(name)  # TODO create cached map

        # ?
        # if not self.load_events[cidx].query():
        #     self.load_module(cidx)

        self.compute_stream.wait_event(self.load_events[cidx])  # pyright: ignore[reportArgumentType]
        with torch.cuda.stream(self.compute_stream):
            output = original_forward(*args, **kwargs)
        self.compute_events[cidx].record()

        if self.window_size > 0:
            nidx = (cidx + self.window_size) % len(self.execution_order)
            self.load_module(nidx)
        pidx = (cidx - 1) % len(self.execution_order)
        self.unload_module(pidx)

        return output
