import torch
import nodes
import inspect
from .libs import utils
from nodes import MAX_RESOLUTION
import logging


class ConcatConditioningsWithMultiplier:
    @classmethod
    def INPUT_TYPES(s):
        stack = inspect.stack()
        if stack[1].function == 'get_input_info':
            # bypass validation
            class AllContainer:
                def __contains__(self, item):
                    return True

                def __getitem__(self, key):
                    return "FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}

            return {
                "required": {"conditioning1": ("CONDITIONING",), },
                "optional": AllContainer()
            }

        return {
            "required": {"conditioning1": ("CONDITIONING",), },
            "optional": {"multiplier1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}), },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/__for_testing"

    def doit(self, **kwargs):
        if "ConditioningMultiplier_PoP" in nodes.NODE_CLASS_MAPPINGS:
            obj = nodes.NODE_CLASS_MAPPINGS["ConditioningMultiplier_PoP"]()
        else:
            utils.try_install_custom_node('https://github.com/picturesonpictures/comfy_PoP',
                                          "To use 'ConcatConditioningsWithMultiplier' node, 'comfy_PoP' extension is required.")
            raise Exception("'comfy_PoP' node isn't installed.")

        conditioning_to = kwargs['conditioning1']
        conditioning_to = obj.multiply_conditioning_strength(conditioning=conditioning_to, multiplier=float(kwargs['multiplier1']))[0]

        out = None
        for k, conditioning_from in kwargs.items():
            if k == 'conditioning1' or not k.startswith('conditioning'):
                continue

            out = []
            if len(conditioning_from) > 1:
                logging.warning(f"[Inspire Pack] ConcatConditioningsWithMultiplier {k} contains more than 1 cond, only the first one will actually be applied to conditioning1.")

            mkey = 'multiplier' + k[12:]
            multiplier = float(kwargs[mkey])
            conditioning_from = obj.multiply_conditioning_strength(conditioning=conditioning_from, multiplier=multiplier)[0]
            cond_from = conditioning_from[0][0]

            for i in range(len(conditioning_to)):
                t1 = conditioning_to[i][0]
                tw = torch.cat((t1, cond_from), 1)
                n = [tw, conditioning_to[i][1].copy()]
                out.append(n)

            conditioning_to = out

        if out is None:
            return (kwargs['conditioning1'],)
        else:
            return (out,)


# CREDIT for ConditioningStretch, ConditioningUpscale: Davemane42
# Imported to support archived custom nodes.
# original code: https://github.com/Davemane42/ComfyUI_Dave_CustomNode/blob/main/MultiAreaConditioning.py
class ConditioningStretch:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "resolutionX": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "resolutionY": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "newWidth": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "newHeight": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                # "scalar": ("INT", {"default": 2, "min": 1, "max": 100, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "InspirePack/conditioning"

    FUNCTION = 'upscale'

    @staticmethod
    def upscale(conditioning, resolutionX, resolutionY, newWidth, newHeight, scalar=1):
        c = []
        for t in conditioning:

            n = [t[0], t[1].copy()]
            if 'area' in n[1]:
                newWidth *= scalar
                newHeight *= scalar

                x = ((n[1]['area'][3] * 8) * newWidth / resolutionX) // 8
                y = ((n[1]['area'][2] * 8) * newHeight / resolutionY) // 8
                w = ((n[1]['area'][1] * 8) * newWidth / resolutionX) // 8
                h = ((n[1]['area'][0] * 8) * newHeight / resolutionY) // 8

                n[1]['area'] = tuple(map(lambda x: (((int(x) + 7) >> 3) << 3), [h, w, y, x]))

            c.append(n)

        return (c,)


class ConditioningUpscale:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "scalar": ("INT", {"default": 2, "min": 1, "max": 100, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "InspirePack/conditioning"

    FUNCTION = 'upscale'

    @staticmethod
    def upscale(conditioning, scalar):
        c = []
        for t in conditioning:

            n = [t[0], t[1].copy()]
            if 'area' in n[1]:
                n[1]['area'] = tuple(map(lambda x: ((x * scalar + 7) >> 3) << 3, n[1]['area']))

            c.append(n)

        return (c,)


NODE_CLASS_MAPPINGS = {
    "ConcatConditioningsWithMultiplier //Inspire": ConcatConditioningsWithMultiplier,
    "ConditioningUpscale //Inspire": ConditioningUpscale,
    "ConditioningStretch //Inspire": ConditioningStretch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConcatConditioningsWithMultiplier //Inspire": "Concat Conditionings with Multiplier (Inspire)",
    "ConditioningUpscale //Inspire": "Conditioning Upscale (Inspire)",
    "ConditioningStretch //Inspire": "Conditioning Stretch (Inspire)",
}
