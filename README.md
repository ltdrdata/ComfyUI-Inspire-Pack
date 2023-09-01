# ComfyUI-Inspire-Pack
This repository offers various extension nodes for ComfyUI. Nodes here have different characteristics compared to those in the ComfyUI Impact Pack. The Impact Pack has become too large now...

## Nodes
* Lora Block Weight - This is a node that provides functionality related to Lora block weight.
    * This provides similar functionality to [sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)
    * `Lora Loader (Block Weight)`: When loading Lora, the block weight vector is applied.
        * In the block vector, you can use numbers, R, A, a, B, and b.
        * R is determined sequentially based on a random seed, while A and B represent the values of the A and B parameters, respectively. a and b are half of the values of A and B, respectively.
    * `XY Input: Lora Block Weight`: This is a node in the [Efficiency Nodes](https://github.com/LucianoCirino/efficiency-nodes-comfyui)' XY Plot that allows you to use Lora block weight.
        * You must ensure that X and Y connections are made, and dependencies should be connected to the XY Plot.
        * NOTE: It is currently unavailable and can only be used after [PR/111](https://github.com/LucianoCirino/efficiency-nodes-comfyui/pull/111) has been merged.

* SEGS Supports nodes - This is a node that supports ApplyControlNet (SEGS) from the Impact Pack.
    * `OpenPose Preprocessor Provider (SEGS)`: OpenPose preprocessor is applied for the purpose of using OpenPose ControlNet in SEGS.
        * You need to install [ControlNet Auxiliary Preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux) to use this.
    * `Canny Preprocessor Provider (SEGS)`: Canny preprocessor is applied for the purpose of using Canny ControlNet in SEGS.