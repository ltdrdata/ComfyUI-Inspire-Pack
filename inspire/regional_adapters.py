# The adapter families that Inspire implements itself, each as a helper and a regional node for Apply Regional Adapters.
# append_tokens grows the attention mask as StyleModelApply does: ComfyUI nodes.py (a20552b, GPL-3.0).
import math

import folder_paths
import torch

from .model_nodes import adapter_families, load_adapter_models
from .regional_nodes import RegionalAdapter


class ReduxModelHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_model_file": (folder_paths.get_filename_list("style_models"),),
                "clip_vision_file": (folder_paths.get_filename_list("clip_vision"),),
                "cache_mode": (list(adapter_families["Redux"]["cache_modes"]), {"default": "none"}),
            },
        }

    RETURN_TYPES = ("STYLE_MODEL", "CLIP_VISION")
    RETURN_NAMES = ("style_model", "clip_vision")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/models"

    @staticmethod
    def doit(style_model_file, clip_vision_file, cache_mode):
        return load_adapter_models("ReduxModelHelper", "Redux", cache_mode, style_model_name=style_model_file, clip_name=clip_vision_file)


BLOCKED = math.log(1e-6)  # finite: core interpolates the mask when the token grid of the latent differs from the mask's


def region_bias(mask, grid):
    """mask [M, H, W] -> [grid tokens]: the log of the share of each token that the region covers, at least BLOCKED."""
    share = torch.nn.functional.interpolate(mask.reshape(-1, 1, *mask.shape[-2:])[:1].float(), size=grid, mode="area")
    return share.flatten().clamp(min=1e-6).log()


def append_tokens(txt, keys, tokens, mask):
    """Appends the tokens to one conditioning entry and grows its attention_mask by quadrants. With a mask, the image tokens
    attend to the new tokens by the share of the region they cover, the earlier tokens do not attend to them, and the new
    tokens do not attend to other regions' tokens."""
    keys = keys.copy()
    out = torch.cat((txt, tokens.to(txt)), dim=1)
    old = keys.get("attention_mask")
    if mask is None and old is None:
        return [out, keys]

    n_txt, n = txt.shape[1], tokens.shape[1]
    grid = keys.get("attention_mask_img_shape", (math.ceil(mask.shape[-2] / 16), math.ceil(mask.shape[-1] / 16)) if mask is not None else (1, 1))
    n_img = grid[0] * grid[1]
    if old is None:
        old = torch.zeros((txt.shape[0], n_txt + n_img, n_txt + n_img), dtype=torch.float16)
    elif old.dtype == torch.bool:
        old = torch.log(old.to(torch.float16))
    new = torch.zeros((txt.shape[0], n_txt + n + n_img, n_txt + n + n_img), dtype=torch.float16)
    new[:, :n_txt, :n_txt] = old[:, :n_txt, :n_txt]
    new[:, :n_txt, n_txt + n:] = old[:, :n_txt, n_txt:]
    new[:, n_txt + n:, :n_txt] = old[:, n_txt:, :n_txt]
    new[:, n_txt + n:, n_txt + n:] = old[:, n_txt:, n_txt:]
    if mask is not None:
        new[:, n_txt:n_txt + n, :n_txt] = new[:, :1, :n_txt]
        new[:, :n_txt, n_txt:n_txt + n] = BLOCKED
        new[:, n_txt + n:, n_txt:n_txt + n] = region_bias(mask, grid)[None, :, None].to(new.dtype)
    keys["attention_mask"] = new
    keys["attention_mask_img_shape"] = grid
    return [out, keys]


class RegionalRedux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_model": ("STYLE_MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("REGIONAL_ADAPTER", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(style_model, clip_vision, image, weight, start_at, end_at, mask=None):
        # as CLIP Vision Encode (crop center) and Apply Style Model (multiply): the tokens of every picture of `image`
        tokens = style_model.get_cond(clip_vision.encode_image(image, crop=True)).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0) * weight

        def pieces(item):
            # the range splits the entry into the timestep pieces of Conditioning Set Timestep Range; only the piece inside it gets the tokens
            txt, keys = item
            if txt.shape[-1] != tokens.shape[-1]:
                raise Exception(f"[Inspire Pack] Regional Redux: the style model is not a Redux model for this base model "
                                f"(its tokens are {tokens.shape[-1]} wide, those of the prompt {txt.shape[-1]}).")
            start, end = keys.get("start_percent", 0.0), keys.get("end_percent", 1.0)
            low, high = max(start, start_at), min(end, end_at)
            if low >= high:
                return [item]
            added = append_tokens(txt, keys, tokens, mask)
            if (low, high) == (start, end):
                return [added]
            added[1].update(start_percent=low, end_percent=high)
            before = [[txt, keys | {"start_percent": start, "end_percent": low}]] if low > start else []
            after = [[txt, keys | {"start_percent": high, "end_percent": end}]] if high < end else []
            return before + [added] + after

        def apply_conditioning(positive, negative):
            return [piece for item in positive for piece in pieces(item)], negative

        return (RegionalAdapter(mask, apply_conditioning=apply_conditioning), )


NODE_CLASS_MAPPINGS = {
    "ReduxModelHelper //Inspire": ReduxModelHelper,
    "RegionalRedux //Inspire": RegionalRedux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReduxModelHelper //Inspire": "Redux Model Helper (Inspire)",
    "RegionalRedux //Inspire": "Regional Redux (Inspire)",
}
