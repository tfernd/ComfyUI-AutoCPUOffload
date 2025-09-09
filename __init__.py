from .offload import AutoCPUOffloadManager, clear_model_patches

from comfy.model_patcher import ModelPatcher


class OffloadedModelPatcher(ModelPatcher):
    patch_on_device = False

    def __init__(self, *args, manager: AutoCPUOffloadManager, **kwargs):
        super().__init__(*args, **kwargs)

        self.manager = manager

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
    ):
        super().load(
            device_to=device_to,
            lowvram_model_memory=0,
            force_patch_weights=True,
            full_load=False,
        )

        self.manager.initialize()
        self.manager.set_pin_memory()

    def detach(self, unpatch_all=True):
        super().detach(unpatch_all=unpatch_all)
        clear_model_patches(self.model)

    # def patch_weight_to_device() # TODO


class AutoCPUOffload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "window_size": (
                    "INT",
                    {"default": 8, "min": 1, "max": 1_024, "step": 1},
                ),
                "enable": ("BOOLEAN", {"default": True}),
                "pin_memory": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "INFO")
    RETURN_NAMES = ("model", "info")

    FUNCTION = "patch"
    OUTPUT_NODE = False

    CATEGORY = "patches"

    def patch(
        self,
        model: ModelPatcher,
        window_size: int,
        enable: bool,
        pin_memory: bool,
        # seed: int,
    ):
        patcher: ModelPatcher = getattr(model, "patcher", model)
        diffusion_model = patcher.model.diffusion_model

        info = None
        if enable:
            manager = AutoCPUOffloadManager(
                diffusion_model,
                window_size=window_size,
                pin_memory=pin_memory,
                device=patcher.load_device,
            )

            # TODO clone patcher...
            patcher = OffloadedModelPatcher(
                patcher.model,
                load_device=patcher.load_device,
                offload_device=patcher.offload_device,
                manager=manager,
            )

            info = dict(
                num_leaf_modules=manager.num_leaf_modules,
                # leaf_module_names=manager.leaf_module_names,
            )
        else:
            clear_model_patches(diffusion_model)

        return (patcher, info)

    @classmethod
    def IS_CHANGED(cls, model, window_size, enable, pin_memory):
        return True


NODE_CLASS_MAPPINGS = {"AutoCPUOffload": AutoCPUOffload}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoCPUOffload": "Auto CPU Offload"}
