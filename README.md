# Auto CPU Offload for ComfyUI

**⚠️ WORK IN PROGRESS & POTENTIAL MEMORY LEAKS ⚠️**

This is an **experimental** extension for ComfyUI. It is currently under active development and may be unstable. Users should be aware that there are known memory leaks, which can lead to increased memory consumption over time. Please use with caution and restart ComfyUI if you experience performance degradation.

## About

This extension introduces an "Auto CPU Offload" node designed to reduce GPU VRAM usage by automatically offloading model components to the CPU. It intelligently manages the movement of model layers between the GPU and CPU, aiming to keep only the necessary parts in VRAM during inference.

This is achieved by tracing the model's execution order on the first run and then using this trace to preemptively load and unload layers in a pipelined fashion on subsequent runs. This can allow for the use of larger models or higher resolution workflows that would otherwise exceed available VRAM.

## Features

- **Automatic Offloading:** Intelligently moves model layers not in immediate use to the CPU.
- **Execution Tracing:** Traces the model's execution path to optimize the loading and unloading of modules.
- **Pipelined Memory Management:** Pre-fetches upcoming model layers to the GPU while offloading those that have already been used.
- **Configurable Window Size:** Allows you to set the number of upcoming layers to pre-load, which can be tuned for your specific hardware.
- **Pin Memory Support:** Optionally pins memory for faster data transfer between CPU and GPU.

## Installation

1.  Navigate to your `ComfyUI/custom_nodes/` directory.
2.  Clone this repository:
    <!-- ```bash
    git clone <your-repository-url>
    ``` -->
3.  Restart ComfyUI.

## Usage

1.  In your ComfyUI workflow, add the "Auto CPU Offload" node, which can be found in the "patches" category.
2.  Connect your model loader to the `model` input of the "Auto CPU Offload" node.
3.  Connect the `model` output of the "Auto CPU Offload" node to the next node in your workflow that requires the model.
4.  Adjust the following parameters on the node as needed:
    - **`enable`**: A boolean to toggle the offloading functionality on or off.
    - **`window_size`**: An integer that determines how many subsequent layers are pre-loaded into VRAM. A larger window size may lead to smoother performance at the cost of higher VRAM usage.
    - **`pin_memory`**: A boolean that, when enabled, can speed up the transfer of data between the CPU and GPU.

## Known Issues

- **Memory Leaks:** There are known memory leaks in the current version. This can cause a gradual increase in memory usage over time, potentially leading to performance issues or crashes. It is recommended to monitor your memory usage and restart ComfyUI periodically.

## Future Work

- Resolving the identified memory leaks.
- Improving the efficiency of the offloading process.
- Adding more detailed performance metrics and debugging information.

## Contributing

As this is a work in progress, contributions and feedback are welcome. Please feel free to open an issue or submit a pull request if you have suggestions for improvements.

## License

See the `LICENSE` file for details.
