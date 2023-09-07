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
        * Note: To use this feature, update `Efficient Nodes` to a version released after September 3rd.

* SEGS Supports nodes - This is a node that supports ApplyControlNet (SEGS) from the Impact Pack.
    * `OpenPose Preprocessor Provider (SEGS)`: OpenPose preprocessor is applied for the purpose of using OpenPose ControlNet in SEGS.
        * You need to install [ControlNet Auxiliary Preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux) to use this.
    * `Canny Preprocessor Provider (SEGS)`: Canny preprocessor is applied for the purpose of using Canny ControlNet in SEGS.

* A1111 Compatibility support - These nodes assists in replicating the creation of A1111 in ComfyUI exactly.
    * `KSampler (Inspire)`: ComfyUI uses the CPU for generating random noise, while A1111 uses the GPU. One of the three factors that significantly impact reproducing A1111's results in ComfyUI can be addressed using `KSampler (Inspire)`.
        * Other point #1 : Please make sure you haven't forgotten to include 'embedding:' in the embedding used in the prompt, like 'embedding:easynegative.'
        * Other point #2 : ComfyUI and A1111 have different interpretations of weighting. To align them, you need to use [BlenderNeko/Advanced CLIP Text Encode](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb).

* Prompt Support - These are nodes for supporting prompt processing.
  * `Load Prompts From File (Inspire)`: It sequentially reads prompt files from the specified directory. The output it returns is ZIPPED_PROMPT.
    * Specify the directories located under `ComfyUI-Inspire-Pack/prompts/`
  * `Unzip Prompt (Inspire)`: Separate ZIPPED_PROMPT into `positive`, `negative`, and name components. 
    * `positive` and `negative` represent text prompts, while `name` represents the name of the prompt. When loaded from a file using `Load Prompts From File (Inspire)`, the name corresponds to the file name.
  * `Zip Prompt (Inspire)`: Create ZIPPED_PROMPT from positive, negative, and name_opt.
    * If name_opt is omitted, it will be considered as an empty name.


## Credits

ComfyUI/[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

ComfyUI/[sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight) - The original idea for LoraBlockWeight came from here, and it is based on the syntax of this extension.

LucianoCirino[efficiency-nodes-comfyui](https://github.com/LucianoCirino/efficiency-nodes-comfyui) - The `XY Input` provided by the Inspire Pack supports the `XY Plot` of this node.

Fannovel16/[comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) - The wrapper for the controlnet preprocessor in the Inspire Pack depends on these nodes.

