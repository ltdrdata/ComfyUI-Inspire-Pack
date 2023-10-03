import comfy
import nodes
import numpy as np
import torch

class RegionalPromptSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
                "mask": ("MASK",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "wildcard_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "wildcard prompt"}),
            },
        }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "Inspire/RegionalSampler"

    def doit(self, basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt):
        if 'RegionalPrompt' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use RegionalPromptSimple, you need to install 'ComfyUI-Impact-Pack'")

        model, clip, vae, positive, negative = basic_pipe

        iwe = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode']()
        kap = nodes.NODE_CLASS_MAPPINGS['KSamplerAdvancedProvider']()
        rp = nodes.NODE_CLASS_MAPPINGS['RegionalPrompt']()

        model, clip, positive, _ = iwe.doit(model=model, clip=clip, populated_text=wildcard_prompt)
        basic_pipe = model, clip, vae, positive, negative

        sampler = kap.doit(cfg, sampler_name, scheduler, basic_pipe)[0]
        regional_prompts = rp.doit(mask, sampler)[0]

        return (regional_prompts, )


class RegionalPromptColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "wildcard_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "wildcard prompt"}),
            },
        }

    RETURN_TYPES = ("REGIONAL_PROMPTS", "MASK")
    FUNCTION = "doit"

    CATEGORY = "Inspire/RegionalSampler"

    def doit(self, basic_pipe, color_mask, mask_color, cfg, sampler_name, scheduler, wildcard_prompt):
        try:
            if mask_color.startswith("#"):
                selected = int(mask_color[1:], 16)
            else:
                selected = int(mask_color, 10)
        except Exception:
            raise Exception(f"[ERROR] Invalid mask_color value. mask_color should be color value for RGB")

        red = (selected >> 16) & 0xFF
        green = (selected >> 8) & 0xFF
        blue = selected & 0xFF

        mask_color = np.array([red, green, blue])
        image = 255. * color_mask.cpu().numpy()
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.array(image).squeeze(0)

        h, w, _ = image.shape

        mask = [
                    [1.0 if np.array_equal(pixel, mask_color) else 0.0 for pixel in row] for row in image
               ]
        mask = torch.tensor(mask)

        rp = RegionalPromptSimple().doit(basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt)[0]
        return (rp, mask)


NODE_CLASS_MAPPINGS = {
    "RegionalPromptSimple //Inspire": RegionalPromptSimple,
    "RegionalPromptColorMask //Inspire": RegionalPromptColorMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionalPromptSimple //Inspire": "Regional Prompt Simple (Inspire)",
    "RegionalPromptColorMask //Inspire": "Regional Prompt By Color Mask (Inspire)",
}
