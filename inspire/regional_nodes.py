import traceback

import comfy
import comfy.utils
import nodes
import torch
import re
import webcolors

from . import prompt_support
from .libs import utils, common

import logging

class RegionalPromptSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
                "mask": ("MASK",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (common.get_schedulers(),),
                "wildcard_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "wildcard prompt"}),
                "controlnet_in_pipe": ("BOOLEAN", {"default": False, "label_on": "Keep", "label_off": "Override"}),
                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variation_method": (["linear", "slerp"],),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
            }
        }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt,
             controlnet_in_pipe=False, sigma_factor=1.0, variation_seed=0, variation_strength=0.0, variation_method='linear', scheduler_func_opt=None):
        if 'RegionalPrompt' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Impact-Pack',
                                          "To use 'RegionalPromptSimple' node, 'Impact Pack' extension is required.")
            raise Exception("[ERROR] To use RegionalPromptSimple, you need to install 'ComfyUI-Impact-Pack'")

        model, clip, vae, positive, negative = basic_pipe

        iwe = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode']()
        kap = nodes.NODE_CLASS_MAPPINGS['KSamplerAdvancedProvider']()
        rp = nodes.NODE_CLASS_MAPPINGS['RegionalPrompt']()

        if wildcard_prompt != "":
            model, clip, new_positive, _ = iwe.doit(model=model, clip=clip, populated_text=wildcard_prompt, seed=None)

            if controlnet_in_pipe:
                prev_cnet = None
                for t in positive:
                    if 'control' in t[1] and 'control_apply_to_uncond' in t[1]:
                        prev_cnet = t[1]['control'], t[1]['control_apply_to_uncond']
                        break

                if prev_cnet is not None:
                    for t in new_positive:
                        t[1]['control'] = prev_cnet[0]
                        t[1]['control_apply_to_uncond'] = prev_cnet[1]

        else:
            new_positive = positive

        basic_pipe = model, clip, vae, new_positive, negative

        sampler = kap.doit(cfg, sampler_name, scheduler, basic_pipe, sigma_factor=sigma_factor, scheduler_func_opt=scheduler_func_opt)[0]
        try:
            regional_prompts = rp.doit(mask, sampler, variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method)[0]
        except:
            raise Exception("[Inspire-Pack] ERROR: Impact Pack is outdated. Update Impact Pack to latest version to use this.")

        return (regional_prompts, )


def color_to_mask(color_mask, mask_color):
    try:
        if mask_color.startswith("#") or mask_color.isalpha():
            hex = mask_color[1:] if mask_color.startswith("#") else webcolors.name_to_hex(mask_color)[1:]
            selected = int(hex, 16)
        else:
            selected = int(mask_color, 10)
    except Exception:
        raise Exception("[ERROR] Invalid mask_color value. mask_color should be a color value for RGB")

    temp = (torch.clamp(color_mask, 0, 1.0) * 255.0).round().to(torch.int)
    temp = torch.bitwise_left_shift(temp[:, :, :, 0], 16) + torch.bitwise_left_shift(temp[:, :, :, 1], 8) + temp[:, :, :, 2]
    mask = torch.where(temp == selected, 1.0, 0.0)
    return mask


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
                "scheduler": (common.get_schedulers(),),
                "wildcard_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "wildcard prompt"}),
                "controlnet_in_pipe": ("BOOLEAN", {"default": False, "label_on": "Keep", "label_off": "Override"}),
                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variation_method": (["linear", "slerp"],),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
            }
        }

    RETURN_TYPES = ("REGIONAL_PROMPTS", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(basic_pipe, color_mask, mask_color, cfg, sampler_name, scheduler, wildcard_prompt,
             controlnet_in_pipe=False, sigma_factor=1.0, variation_seed=0, variation_strength=0.0, variation_method="linear", scheduler_func_opt=None):
        mask = color_to_mask(color_mask, mask_color)
        rp = RegionalPromptSimple().doit(basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt, controlnet_in_pipe,
                                         sigma_factor=sigma_factor, variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method, scheduler_func_opt=scheduler_func_opt)[0]
        return rp, mask


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

    @staticmethod
    def doit(clip, mask, strength, set_cond_area, prompt):
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
            "optional": {
                "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(clip, color_mask, mask_color, strength, set_cond_area, prompt, dilation=0):
        mask = color_to_mask(color_mask, mask_color)

        if dilation != 0:
            mask = utils.dilate_mask(mask, dilation)

        conditioning = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        conditioning = nodes.ConditioningSetMask().append(conditioning, mask, set_cond_area, strength)[0]
        return conditioning, mask


class ToIPAdapterPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "model": ("MODEL",),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            }
        }

    RETURN_TYPES = ("IPADAPTER_PIPE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    @staticmethod
    def doit(ipadapter, model, clip_vision, insightface=None):
        pipe = ipadapter, model, clip_vision, insightface, lambda x: x

        return (pipe,)


class FromIPAdapterPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_pipe": ("IPADAPTER_PIPE", ),
            }
        }

    RETURN_TYPES = ("IPADAPTER", "MODEL", "CLIP_VISION", "INSIGHTFACE")
    RETURN_NAMES = ("ipadapter", "model", "clip_vision", "insight_face")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, ipadapter_pipe):
        ipadapter, model, clip_vision, insightface, _ = ipadapter_pipe
        return ipadapter, model, clip_vision, insightface


class IPAdapterConditioning:
    def __init__(self, mask, weight, weight_type, noise=None, image=None, neg_image=None, embeds=None, start_at=0.0, end_at=1.0, combine_embeds='concat', unfold_batch=False, weight_v2=False, neg_embeds=None):
        self.mask = mask
        self.image = image
        self.neg_image = neg_image
        self.embeds = embeds
        self.neg_embeds = neg_embeds
        self.weight = weight
        self.noise = noise
        self.weight_type = weight_type
        self.start_at = start_at
        self.end_at = end_at
        self.unfold_batch = unfold_batch
        self.weight_v2 = weight_v2
        self.combine_embeds = combine_embeds

    def doit(self, ipadapter_pipe):
        ipadapter, model, clip_vision, insightface, _ = ipadapter_pipe

        if 'IPAdapterAdvanced' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/cubiq/ComfyUI_IPAdapter_plus',
                                          "To use 'Regional IPAdapter' node, 'ComfyUI IPAdapter Plus' extension is required.")
            raise Exception("[ERROR] To use IPAdapterModelHelper, you need to install 'ComfyUI IPAdapter Plus'")

        if self.embeds is None:
            obj = nodes.NODE_CLASS_MAPPINGS['IPAdapterAdvanced']
            model = obj().apply_ipadapter(model=model, ipadapter=ipadapter, weight=self.weight, weight_type=self.weight_type,
                                          start_at=self.start_at, end_at=self.end_at, combine_embeds=self.combine_embeds,
                                          clip_vision=clip_vision, image=self.image, image_negative=self.neg_image, attn_mask=self.mask,
                                          insightface=insightface, weight_faceidv2=self.weight_v2)[0]
        else:
            obj = nodes.NODE_CLASS_MAPPINGS['IPAdapterEmbeds']
            model = obj().apply_ipadapter(model=model, ipadapter=ipadapter, pos_embed=self.embeds, weight=self.weight, weight_type=self.weight_type,
                                          start_at=self.start_at, end_at=self.end_at, neg_embed=self.neg_embeds,
                                          attn_mask=self.mask, clip_vision=clip_vision)[0]

        return model


IPADAPTER_WEIGHT_TYPES_CACHE = None


def IPADAPTER_WEIGHT_TYPES():
    global IPADAPTER_WEIGHT_TYPES_CACHE

    if IPADAPTER_WEIGHT_TYPES_CACHE is None:
        try:
            IPADAPTER_WEIGHT_TYPES_CACHE = nodes.NODE_CLASS_MAPPINGS['IPAdapterAdvanced']().INPUT_TYPES()['required']['weight_type'][0]
        except Exception:
            logging.error("[Inspire Pack] IPAdapterPlus is not installed.")
            IPADAPTER_WEIGHT_TYPES_CACHE = ["IPAdapterPlus is not installed"]

    return IPADAPTER_WEIGHT_TYPES_CACHE


class RegionalIPAdapterMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),

                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (IPADAPTER_WEIGHT_TYPES(), ),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "faceid_v2": ("BOOLEAN", {"default": False}),
                "weight_v2": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "neg_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(mask, image, weight, noise, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, faceid_v2=False, weight_v2=False, combine_embeds="concat", neg_image=None):
        cond = IPAdapterConditioning(mask, weight, weight_type, noise=noise, image=image, neg_image=neg_image, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, weight_v2=weight_v2, combine_embeds=combine_embeds)
        return (cond, )


class RegionalIPAdapterColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (IPADAPTER_WEIGHT_TYPES(), ),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "faceid_v2": ("BOOLEAN", {"default": False }),
                "weight_v2": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "neg_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(color_mask, mask_color, image, weight, noise, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, faceid_v2=False, weight_v2=False, combine_embeds="concat", neg_image=None):
        mask = color_to_mask(color_mask, mask_color)
        cond = IPAdapterConditioning(mask, weight, weight_type, noise=noise, image=image, neg_image=neg_image, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, weight_v2=weight_v2, combine_embeds=combine_embeds)
        return (cond, mask)


class RegionalIPAdapterEncodedMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),

                "embeds": ("EMBEDS",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "weight_type": (IPADAPTER_WEIGHT_TYPES(), ),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "neg_embeds": ("EMBEDS",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(mask, embeds, weight, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, neg_embeds=None):
        cond = IPAdapterConditioning(mask, weight, weight_type, embeds=embeds, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, neg_embeds=neg_embeds)
        return (cond, )


class RegionalIPAdapterEncodedColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),

                "embeds": ("EMBEDS",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "weight_type": (IPADAPTER_WEIGHT_TYPES(), ),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "neg_embeds": ("EMBEDS",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(color_mask, mask_color, embeds, weight, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, neg_embeds=None):
        mask = color_to_mask(color_mask, mask_color)
        cond = IPAdapterConditioning(mask, weight, weight_type, embeds=embeds, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, neg_embeds=neg_embeds)
        return (cond, mask)


class ApplyRegionalIPAdapters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ipadapter_pipe": ("IPADAPTER_PIPE",),
                    "regional_ipadapter1": ("REGIONAL_IPADAPTER", ),
                    },
                }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(**kwargs):
        ipadapter_pipe = kwargs['ipadapter_pipe']
        ipadapter, model, clip_vision, insightface, lora_loader = ipadapter_pipe

        del kwargs['ipadapter_pipe']

        for k, v in kwargs.items():
            ipadapter_pipe = ipadapter, model, clip_vision, insightface, lora_loader
            model = v.doit(ipadapter_pipe)

        return (model, )


class RegionalAdapter:
    """A REGIONAL_ADAPTER entry: `apply_model(model) -> model`, `apply_conditioning(positive, negative) -> (positive, negative)`
    that changes the prompt itself, and `region_conditioning(positive, negative) -> (positive, negative)` that gives the region
    its own copies of the prompt, the positive one limited to `mask`. Each part may be None.
    Without a mask the entry covers the whole picture."""
    def __init__(self, mask, apply_model=None, apply_conditioning=None, region_conditioning=None):
        self.mask = mask
        self.apply_model = apply_model
        self.apply_conditioning = apply_conditioning
        self.region_conditioning = region_conditioning


def uncovered_area(masks):
    # the sampler stretches every mask to the latent, so masks of other sizes are stretched to one size here
    size = max(mask.shape[-2] for mask in masks), max(mask.shape[-1] for mask in masks)
    rest = torch.ones(size)
    for mask in masks:
        rest = rest - torch.nn.functional.interpolate(mask.reshape(-1, 1, *mask.shape[-2:]), size=size, mode="bilinear").squeeze(1)
    return torch.clamp(rest, 0.0, 1.0)


class ApplyRegionalAdapters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    },
                "optional": {
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "regional_adapter1": ("REGIONAL_ADAPTER", ),
                    },
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("MODEL", "positive", "negative")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(model, positive=None, negative=None, **kwargs):
        prompting = [name for name, x in kwargs.items() if x.apply_conditioning is not None or x.region_conditioning is not None]
        if prompting and (positive is None or negative is None):
            raise Exception(f"[Inspire Pack] Apply Regional Adapters: {prompting[0]} needs `positive` and `negative` linked, "
                            "because it changes the prompt.")

        for regional_adapter in kwargs.values():
            if regional_adapter.apply_model is not None:
                model = regional_adapter.apply_model(model)
            if regional_adapter.apply_conditioning is not None:
                positive, negative = regional_adapter.apply_conditioning(positive, negative)

        copying = [x for x in kwargs.values() if x.region_conditioning is not None]
        if not copying:
            return model, positive, negative

        positives, negatives = [], []
        for regional_adapter in copying:
            region_positive, region_negative = regional_adapter.region_conditioning(positive, negative)
            positives += region_positive
            negatives += region_negative

        # without the plain prompt on the area that no region covers, the whole picture turns grey;
        # a copy without a mask covers the whole picture
        masks = [x.mask for x in copying]
        if all(mask is not None for mask in masks):
            rest = uncovered_area(masks)
            if rest.any():
                positives += nodes.ConditioningSetMask().append(positive, rest, "default", 1.0)[0]

        return model, positives, negatives


def require_face(family, face_analysis, image, stop):
    """Stops when no picture of `image` shows a face at the detector sizes the family's pack tries: 640 down to `stop`, in steps of 64."""
    for picture in image.mul(255).clamp(0, 255).byte().cpu()[..., [2, 1, 0]].numpy():
        for size in range(640, stop, -64):
            face_analysis.det_model.input_size = (size, size)
            if face_analysis.get(picture):
                return
    raise Exception(f"[Inspire Pack] Regional {family}: no face was found in the reference image.")


class RegionalPulid:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pulid": ("PULID",),
                "eva_clip": ("EVA_CLIP",),
                "face_analysis": ("FACEANALYSIS",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "method": (["fidelity", "style", "neutral"],),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("REGIONAL_ADAPTER", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method, mask=None):
        require_face("PuLID", face_analysis, image, 256)

        def apply_model(model):
            return nodes.NODE_CLASS_MAPPINGS['ApplyPulid']().apply_pulid(model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at,
                                                                        method=method, attn_mask=mask)[0]

        return (RegionalAdapter(mask, apply_model=apply_model), )


def keypoints_picture(image, mask):
    """The first reference on a black picture of the mask's size, fitted to the mask's bounding box, centred across it
    and with its centre at a third of its height."""
    mask = mask.reshape(-1, *mask.shape[-2:])[0]
    picture = torch.zeros((1, *mask.shape, 3))
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(ys) == 0:
        return picture

    top, left = int(ys.min()), int(xs.min())
    box_height, box_width = int(ys.max()) + 1 - top, int(xs.max()) + 1 - left
    scale = min(box_width / image.shape[2], box_height / image.shape[1])
    width, height = round(image.shape[2] * scale), round(image.shape[1] * scale)
    x = left + (box_width - width) // 2
    y = top + min(max(round(box_height / 3 - height / 2), 0), box_height - height)
    picture[:, y:y + height, x:x + width] = comfy.utils.common_upscale(image[:1].movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
    return picture


class RegionalInstantID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid": ("INSTANTID",),
                "insightface": ("FACEANALYSIS",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "controlnet_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "mask": ("MASK",),
                "image_kps": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("REGIONAL_ADAPTER", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(instantid, insightface, control_net, image, weight, start_at, end_at, controlnet_strength, noise, mask=None, image_kps=None):
        require_face("InstantID", insightface, image, 128)

        def apply_model(model):
            return nodes.NODE_CLASS_MAPPINGS['InstantIDAttentionPatch']().patch_attention(instantid, insightface, image, model, weight, start_at, end_at,
                                                                                        noise=noise, mask=mask)[0]

        def region_conditioning(positive, negative):
            # weight 0 leaves the model alone and only gives the face embeds
            face_embeds = nodes.NODE_CLASS_MAPPINGS['InstantIDAttentionPatch']().patch_attention(instantid, insightface, image, None, 0.0, start_at, end_at,
                                                                                               noise=noise)[1]
            kps = image_kps
            if kps is None:
                # without a mask, the reference itself, as Apply InstantID takes it
                picture = image[:1] if mask is None else keypoints_picture(image, mask)
                kps = nodes.NODE_CLASS_MAPPINGS['FaceKeypointsPreprocessor']().preprocess_image(insightface, picture)[0]
            return nodes.NODE_CLASS_MAPPINGS['ApplyInstantIDControlNet']().apply_controlnet(face_embeds, control_net, kps, positive, negative,
                                                                                          controlnet_strength, start_at, end_at, mask=mask)

        return (RegionalAdapter(mask, apply_model=apply_model, region_conditioning=region_conditioning), )


class RegionalSeedExplorerMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),

                "noise": ("NOISE_IMAGE",),
                "seed_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                "enable_additional": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "additional_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "additional_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_mode": (["GPU(=A1111)", "CPU"],),
            },
            "optional":
                {"variation_method": (["linear", "slerp"],), }
        }

    RETURN_TYPES = ("NOISE_IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(mask, noise, seed_prompt, enable_additional, additional_seed, additional_strength, noise_mode, variation_method='linear'):
        device = comfy.model_management.get_torch_device()
        noise_device = "cpu" if noise_mode == "CPU" else device

        noise = noise.to(device)
        mask = mask.to(device)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=noise.shape[-2:], mode="bilinear").squeeze(0)

        try:
            seed_prompt = seed_prompt.replace("\n", "")
            items = seed_prompt.strip().split(",")

            if items == ['']:
                items = []

            if enable_additional:
                items.append((additional_seed, additional_strength))

            noise = prompt_support.SeedExplorer.apply_variation(noise, items, noise_device, mask, variation_method=variation_method)
        except Exception:
            logging.error("[Inspire Pack] IGNORED: RegionalSeedExplorerColorMask is failed.")
            traceback.print_exc()

        noise = noise.cpu()
        mask.cpu()
        return (noise,)


class RegionalSeedExplorerColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),

                "noise": ("NOISE_IMAGE",),
                "seed_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                "enable_additional": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "additional_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "additional_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_mode": (["GPU(=A1111)", "CPU"],),
            },
            "optional":
                {"variation_method": (["linear", "slerp"],), }
        }

    RETURN_TYPES = ("NOISE_IMAGE", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(color_mask, mask_color, noise, seed_prompt, enable_additional, additional_seed, additional_strength, noise_mode, variation_method='linear'):
        device = comfy.model_management.get_torch_device()
        noise_device = "cpu" if noise_mode == "CPU" else device

        color_mask = color_mask.to(device)
        noise = noise.to(device)

        mask = color_to_mask(color_mask, mask_color)
        original_mask = mask
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=noise.shape[-2:], mode="bilinear").squeeze(0)

        mask = mask.to(device)

        try:
            seed_prompt = seed_prompt.replace("\n", "")
            items = seed_prompt.strip().split(",")

            if items == ['']:
                items = []

            if enable_additional:
                items.append((additional_seed, additional_strength))

            noise = prompt_support.SeedExplorer.apply_variation(noise, items, noise_device, mask, variation_method=variation_method)
        except Exception:
            logging.error("[Inspire Pack] IGNORED: RegionalSeedExplorerColorMask is failed.")
            traceback.print_exc()

        color_mask.cpu()
        noise = noise.cpu()
        original_mask = original_mask.cpu()
        return (noise, original_mask)


class ColorMaskToDepthMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "spec": ("STRING", {"multiline": True, "default": "#FF0000:1.0\n#000000:1.0"}),
                "base_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "flatten_method": (["override", "sum", "max"],),
            },
        }

    RETURN_TYPES = ("MASK", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, color_mask, spec, base_value, dilation, flatten_method):
        specs = spec.split('\n')
        pat = re.compile("(?P<color_code>#[A-F0-9]+):(?P<cfg>[0-9]+(.[0-9]*)?)")

        masks = [torch.ones((1, color_mask.shape[1], color_mask.shape[2])) * base_value]
        for x in specs:
            match = pat.match(x)
            if match:
                mask = color_to_mask(color_mask=color_mask, mask_color=match['color_code']) * float(match['cfg'])
                mask = utils.dilate_mask(mask, dilation)
                masks.append(mask)

        if masks:
            masks = torch.cat(masks, dim=0)
            if flatten_method == 'override':
                masks = utils.flatten_non_zero_override(masks)
            elif flatten_method == 'max':
                masks = torch.max(masks, dim=0)[0]
            else:  # flatten_method == 'sum':
                masks = torch.sum(masks, dim=0)

            masks = torch.clamp(masks, min=0.0, max=1.0)
            masks = masks.unsqueeze(0)
        else:
            masks = torch.tensor([])

        return (masks, )


class RegionalCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "mask": ("MASK",),
                             }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(model, mask):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)

        size = None

        def regional_cfg(args):
            nonlocal mask
            nonlocal size

            x = args['input']

            if mask.device != x.device:
                mask = mask.to(x.device)

            if size != x.shape[-2:]:
                size = x.shape[-2:]
                mask = torch.nn.functional.interpolate(mask, size=size, mode='bilinear', align_corners=False)

            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]

            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * mask

            return x - cfg_result

        m = model.clone()
        m.set_model_sampler_cfg_function(regional_cfg)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "RegionalPromptSimple //Inspire": RegionalPromptSimple,
    "RegionalPromptColorMask //Inspire": RegionalPromptColorMask,
    "RegionalConditioningSimple //Inspire": RegionalConditioningSimple,
    "RegionalConditioningColorMask //Inspire": RegionalConditioningColorMask,
    "RegionalIPAdapterMask //Inspire": RegionalIPAdapterMask,
    "RegionalIPAdapterColorMask //Inspire": RegionalIPAdapterColorMask,
    "RegionalIPAdapterEncodedMask //Inspire": RegionalIPAdapterEncodedMask,
    "RegionalIPAdapterEncodedColorMask //Inspire": RegionalIPAdapterEncodedColorMask,
    "RegionalSeedExplorerMask //Inspire": RegionalSeedExplorerMask,
    "RegionalSeedExplorerColorMask //Inspire": RegionalSeedExplorerColorMask,
    "ToIPAdapterPipe //Inspire": ToIPAdapterPipe,
    "FromIPAdapterPipe //Inspire": FromIPAdapterPipe,
    "ApplyRegionalIPAdapters //Inspire": ApplyRegionalIPAdapters,
    "ApplyRegionalAdapters //Inspire": ApplyRegionalAdapters,
    "RegionalPulid //Inspire": RegionalPulid,
    "RegionalInstantID //Inspire": RegionalInstantID,
    "RegionalCFG //Inspire": RegionalCFG,
    "ColorMaskToDepthMask //Inspire": ColorMaskToDepthMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionalPromptSimple //Inspire": "Regional Prompt Simple (Inspire)",
    "RegionalPromptColorMask //Inspire": "Regional Prompt By Color Mask (Inspire)",
    "RegionalConditioningSimple //Inspire": "Regional Conditioning Simple (Inspire)",
    "RegionalConditioningColorMask //Inspire": "Regional Conditioning By Color Mask (Inspire)",
    "RegionalIPAdapterMask //Inspire": "Regional IPAdapter Mask (Inspire)",
    "RegionalIPAdapterColorMask //Inspire": "Regional IPAdapter By Color Mask (Inspire)",
    "RegionalIPAdapterEncodedMask //Inspire": "Regional IPAdapter Encoded Mask (Inspire)",
    "RegionalIPAdapterEncodedColorMask //Inspire": "Regional IPAdapter Encoded By Color Mask (Inspire)",
    "RegionalSeedExplorerMask //Inspire": "Regional Seed Explorer By Mask (Inspire)",
    "RegionalSeedExplorerColorMask //Inspire": "Regional Seed Explorer By Color Mask (Inspire)",
    "ToIPAdapterPipe //Inspire": "ToIPAdapterPipe (Inspire)",
    "FromIPAdapterPipe //Inspire": "FromIPAdapterPipe (Inspire)",
    "ApplyRegionalIPAdapters //Inspire": "Apply Regional IPAdapters (Inspire)",
    "ApplyRegionalAdapters //Inspire": "Apply Regional Adapters (Inspire)",
    "RegionalPulid //Inspire": "Regional PuLID (Inspire)",
    "RegionalInstantID //Inspire": "Regional InstantID (Inspire)",
    "RegionalCFG //Inspire": "Regional CFG (Inspire)",
    "ColorMaskToDepthMask //Inspire": "Color Mask To Depth Mask (Inspire)",
}
