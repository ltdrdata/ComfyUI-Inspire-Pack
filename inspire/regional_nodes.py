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

    CATEGORY = "InspirePack/Regional"

    def doit(self, basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt):
        if 'RegionalPrompt' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use RegionalPromptSimple, you need to install 'ComfyUI-Impact-Pack'")

        model, clip, vae, positive, negative = basic_pipe

        iwe = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode']()
        kap = nodes.NODE_CLASS_MAPPINGS['KSamplerAdvancedProvider']()
        rp = nodes.NODE_CLASS_MAPPINGS['RegionalPrompt']()

        if wildcard_prompt != "":
            model, clip, positive, _ = iwe.doit(model=model, clip=clip, populated_text=wildcard_prompt)

        basic_pipe = model, clip, vae, positive, negative

        sampler = kap.doit(cfg, sampler_name, scheduler, basic_pipe)[0]
        regional_prompts = rp.doit(mask, sampler)[0]

        return (regional_prompts, )


def color_to_mask(color_mask, mask_color):
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
    return torch.tensor(mask).unsqueeze(0)


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

    CATEGORY = "InspirePack/Regional"

    def doit(self, basic_pipe, color_mask, mask_color, cfg, sampler_name, scheduler, wildcard_prompt):
        mask = color_to_mask(color_mask, mask_color)
        rp = RegionalPromptSimple().doit(basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt)[0]
        return (rp, mask)


class RegionalConditioningSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "mask": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
                "prompt": ("STRING", {"multiline": True, "placeholder": "prompt"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, clip, mask, strength, set_cond_area, prompt):
        conditioning = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        conditioning = nodes.ConditioningSetMask().append(conditioning, mask, set_cond_area, strength)[0]
        return (conditioning, )


class RegionalConditioningColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
                "prompt": ("STRING", {"multiline": True, "placeholder": "prompt"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, clip, color_mask, mask_color, strength, set_cond_area, prompt):
        mask = color_to_mask(color_mask, mask_color)

        conditioning = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        conditioning = nodes.ConditioningSetMask().append(conditioning, mask, set_cond_area, strength)[0]
        return (conditioning, mask)


class ToIPAdapterPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "model": ("MODEL", ),
            }
        }

    RETURN_TYPES = ("IPADAPTER_PIPE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, ipadapter, clip_vision, model):
        pipe = ipadapter, clip_vision, model

        return (pipe,)


class FromIPAdapterPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_pipe": ("IPADAPTER_PIPE", ),
            }
        }

    RETURN_TYPES = ("IPADAPTER", "CLIP_VISION", "MODEL")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, ipadapter_pipe):
        return ipadapter_pipe


class RegionalIPAdapterMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_pipe": ("IPADAPTER_PIPE",),
                "mask": ("MASK",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (["original", "linear", "channel penalty"],),
            },
        }

    RETURN_TYPES = ("IPADAPTER_PIPE", "MODEL")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, ipadapter_pipe, mask, image, weight, noise, weight_type):
        if 'IPAdapterApply' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use RegionalIPAdapterMask, you need to install 'ComfyUI_IPAdapter_plus'")

        obj = nodes.NODE_CLASS_MAPPINGS['IPAdapterApply']

        ipadapter, clip_vision, model = ipadapter_pipe
        model = obj().apply_ipadapter(ipadapter, model, weight, clip_vision=clip_vision, image=image, weight_type=weight_type, noise=noise, attn_mask=mask)[0]

        new_ipadapter_pipe = ipadapter, clip_vision, model
        return new_ipadapter_pipe, model


class RegionalIPAdapterColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_pipe": ("IPADAPTER_PIPE", ),

                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (["original", "linear", "channel penalty"], ),
            },
        }

    RETURN_TYPES = ("IPADAPTER_PIPE", "MODEL", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, ipadapter_pipe, color_mask, mask_color, image, weight, noise, weight_type):
        mask = color_to_mask(color_mask, mask_color)

        if 'IPAdapterApply' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use RegionalIPAdapterColorMask, you need to install 'ComfyUI_IPAdapter_plus'")
        
        obj = nodes.NODE_CLASS_MAPPINGS['IPAdapterApply']

        ipadapter, clip_vision, model = ipadapter_pipe
        model = obj().apply_ipadapter(ipadapter, model, weight, clip_vision=clip_vision, image=image, weight_type=weight_type, noise=noise, attn_mask=mask)[0]

        new_ipadapter_pipe = ipadapter, clip_vision, model
        return new_ipadapter_pipe, model, mask
        

NODE_CLASS_MAPPINGS = {
    "RegionalPromptSimple //Inspire": RegionalPromptSimple,
    "RegionalPromptColorMask //Inspire": RegionalPromptColorMask,
    "RegionalConditioningSimple //Inspire": RegionalConditioningSimple,
    "RegionalConditioningColorMask //Inspire": RegionalConditioningColorMask,
    "RegionalIPAdapterMask //Inspire": RegionalIPAdapterMask,
    "RegionalIPAdapterColorMask //Inspire": RegionalIPAdapterColorMask,
    "ToIPAdapterPipe //Inspire": ToIPAdapterPipe,
    "FromIPAdapterPipe //Inspire": FromIPAdapterPipe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionalPromptSimple //Inspire": "Regional Prompt Simple (Inspire)",
    "RegionalPromptColorMask //Inspire": "Regional Prompt By Color Mask (Inspire)",
    "RegionalConditioningSimple //Inspire": "Regional Conditioning Simple (Inspire)",
    "RegionalConditioningColorMask //Inspire": "Regional Conditioning By Color Mask (Inspire)",
    "RegionalIPAdapterMask //Inspire": "Regional IPAdapter Mask (Inspire)",
    "RegionalIPAdapterColorMask //Inspire": "Regional IPAdapter By Color Mask (Inspire)",
    "ToIPAdapterPipe //Inspire": "ToIPAdapterPipe (Inspire)",
    "FromIPAdapterPipe //Inspire": "FromIPAdapterPipe (Inspire)",
}
