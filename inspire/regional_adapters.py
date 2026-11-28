# The adapter families that Inspire implements itself, each as a helper and a regional node for Apply Regional Adapters.
# append_tokens grows the attention mask as StyleModelApply does: ComfyUI nodes.py (a20552b, GPL-3.0).
# IDFormer, PerceiverAttention, PulidFluxBlock and token_mask: ComfyUI_PuLID_Flux_ll encoders_flux.py, PulidFluxHook.py, pulidflux.py (7c7362b, Apache-2.0).
# id_embedding: the face features of ApplyPulid, PuLID_ComfyUI pulid.py (7c78b56, Apache-2.0).
import logging
import math

import comfy.ldm.common_dit
import comfy.ldm.flux.model
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.patcher_extension
import comfy.utils
import folder_paths
import numpy as np
import torch
import torchvision.transforms.functional as TF
from comfy.ldm.modules.attention import optimized_attention

from .model_nodes import adapter_families, load_adapter_models
from .regional_nodes import RegionalAdapter, require_face

try:
    from facexlib.parsing import init_parsing_model
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
except ImportError:  # comes with PuLID_ComfyUI, which PuLID FLUX Model Helper asks for when it is missing
    init_parsing_model = FaceRestoreHelper = None


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


STATE = "inspire_adapter"


def adapter_state(executor, x, t, c_concat, c_crossattn, control, transformer_options, **kwargs):
    """APPLY_MODEL wrapper: gives the block patches the latent size, the patch size and the sigma of this model call, and
    keeps the token masks they make in it."""
    state = {"latent": tuple(x.shape[-2:]), "patch_size": executor.class_obj.diffusion_model.patch_size, "sigma": float(t.flatten()[0]), "masks": {}}
    return executor(x, t, c_concat, c_crossattn, control, {**transformer_options, STATE: state}, **kwargs)


def add_adapter_state(model):
    if not model.get_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, STATE):
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, STATE, adapter_state)


def token_mask(mask, transformer_options, x):
    """mask [M, H, W] -> [1 or batch, image tokens, 1] on the device and dtype of x, once per model call: the mask resized to
    the latent, padded to the patch size, and the first latent pixel of each patch."""
    state = transformer_options[STATE]
    m = state["masks"].get(id(mask))
    if m is None:
        p = state["patch_size"]
        m = comfy.utils.reshape_mask(mask, (mask.shape[0], 1) + state["latent"])
        m = comfy.ldm.common_dit.pad_to_patch_size(m, (p, p))[:, 0, ::p, ::p].flatten(1).unsqueeze(-1).to(x.device, x.dtype)
        state["masks"][id(mask)] = m
    if m.shape[0] > 1:
        conds = len(transformer_options["cond_or_uncond"])
        m = comfy.utils.repeat_to_batch_size(m, x.shape[0] // conds).repeat(conds, 1, 1)
    return m


def sigma_range(model, start_at, end_at):
    model_sampling = model.get_model_object("model_sampling")
    return model_sampling.percent_to_sigma(start_at), model_sampling.percent_to_sigma(end_at)


def in_range(sigma_range, transformer_options):
    return sigma_range[1] <= transformer_options[STATE]["sigma"] <= sigma_range[0]


def to_device(tensor, device_or_dtype):
    # the patches are also moved to the model's dtype, which these tensors keep
    return tensor.to(device_or_dtype) if tensor is not None and isinstance(device_or_dtype, torch.device) else tensor


def mask_batch(mask):
    return None if mask is None else mask.reshape(-1, *mask.shape[-2:])


def base_name(model):
    return type(model.model.model_config).__name__


def patcher(model):
    return comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())


ops = comfy.ops.manual_cast


def feed_forward(dim, mult=4):
    return torch.nn.Sequential(ops.LayerNorm(dim), ops.Linear(dim, dim * mult, bias=False), torch.nn.GELU(), ops.Linear(dim * mult, dim, bias=False))


def mapping(in_dim, out_dim):
    return torch.nn.Sequential(ops.Linear(in_dim, 1024), ops.LayerNorm(1024), torch.nn.LeakyReLU(), ops.Linear(1024, 1024), ops.LayerNorm(1024),
                               torch.nn.LeakyReLU(), ops.Linear(1024, out_dim))


class PerceiverAttention(torch.nn.Module):
    """The latents attend to x, and with self_kv also to themselves: PerceiverAttention (self_kv) and PerceiverAttentionCA of PuLID."""
    def __init__(self, dim, dim_head, heads, kv_dim, self_kv):
        super().__init__()
        self.heads = heads
        self.self_kv = self_kv
        self.norm1 = ops.LayerNorm(kv_dim)
        self.norm2 = ops.LayerNorm(dim)
        self.to_q = ops.Linear(dim, dim_head * heads, bias=False)
        self.to_kv = ops.Linear(kv_dim, dim_head * heads * 2, bias=False)
        self.to_out = ops.Linear(dim_head * heads, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        k, v = self.to_kv(torch.cat((x, latents), dim=-2) if self.self_kv else x).chunk(2, dim=-1)
        if k.shape[0] != latents.shape[0]:
            k, v = k.expand(latents.shape[0], -1, -1), v.expand(latents.shape[0], -1, -1)
        return self.to_out(optimized_attention(self.to_q(latents), k, v, self.heads))


class IDFormer(torch.nn.Module):
    def __init__(self, dim=1024, depth=10, dim_head=64, heads=16, num_id_token=5, num_queries=32, output_dim=2048, ff_mult=4):
        super().__init__()
        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        self.depth = depth // 5
        self.latents = torch.nn.Parameter(torch.empty(1, num_queries, dim))
        self.proj_out = torch.nn.Parameter(torch.empty(dim, output_dim))
        self.layers = torch.nn.ModuleList([torch.nn.ModuleList([PerceiverAttention(dim, dim_head, heads, dim, True), feed_forward(dim, ff_mult)])
                                           for _ in range(depth)])
        for i in range(5):
            setattr(self, f"mapping_{i}", mapping(1024, dim))
        self.id_embedding_mapping = mapping(1280, dim * num_id_token)

    def forward(self, x, y):
        latents = comfy.ops.cast_to_input(self.latents, x).repeat(x.shape[0], 1, 1)
        x = self.id_embedding_mapping(x).reshape(-1, self.num_id_token, self.dim)
        latents = torch.cat((latents, x), dim=1)
        for i in range(5):
            context = torch.cat((x, getattr(self, f"mapping_{i}")(y[i])), dim=1)
            for attn, ff in self.layers[i * self.depth:(i + 1) * self.depth]:
                latents = attn(context, latents) + latents
                latents = ff(latents) + latents
        return latents[:, :self.num_queries] @ comfy.ops.cast_to_input(self.proj_out, x)


class PulidFlux(torch.nn.Module):
    def __init__(self, num_ca):
        super().__init__()
        self.pulid_encoder = IDFormer()
        self.pulid_ca = torch.nn.ModuleList([PerceiverAttention(3072, 128, 16, 2048, False) for _ in range(num_ca)])


class PulidFluxModels:
    def __init__(self, pulid_file, eva_clip, face_analysis):
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("pulid", pulid_file), safe_load=True)
        model = PulidFlux(len({key.split(".")[1] for key in sd if key.startswith("pulid_ca.")}))
        model.load_state_dict(sd, strict=True, assign=True)
        self.file = pulid_file
        self.pulid = patcher(model)
        self.eva_clip = patcher(eva_clip)
        self.face_analysis = face_analysis


def to_gray(image):
    return (0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]).repeat(1, 3, 1, 1)


def id_embedding(models, image):
    """IMAGE [B, H, W, C] -> the IDFormer output [1, 32, 2048], the mean over the pictures that show a face."""
    device = comfy.model_management.get_torch_device()
    helper = FaceRestoreHelper(upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50", save_ext="png", device="cpu")
    helper.face_parse = init_parsing_model(model_name="bisenet", device="cpu")
    comfy.model_management.load_models_gpu([models.eva_clip, models.pulid])
    eva = models.eva_clip.model
    eva_dtype = next(eva.parameters()).dtype
    embeddings = []
    for i, picture in enumerate(image):
        bgr = np.ascontiguousarray(picture[..., [2, 1, 0]].mul(255).clamp(0, 255).byte().cpu().numpy())
        for size in range(640, 256, -64):
            models.face_analysis.det_model.input_size = (size, size)
            faces = models.face_analysis.get(bgr)
            if faces:
                break
        else:
            logging.warning(f"[Inspire Pack] Regional PuLID FLUX: picture {i} of the reference image shows no face.")
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        helper.clean_all()
        helper.read_image(bgr)
        helper.get_face_landmarks_5(only_center_face=True)
        helper.align_warp_face()
        if not helper.cropped_faces:
            logging.warning(f"[Inspire Pack] Regional PuLID FLUX: facexlib found no face to align in picture {i} of the reference image.")
            continue
        align = torch.from_numpy(helper.cropped_faces[0][..., [2, 1, 0]].copy()).float().div(255).permute(2, 0, 1).unsqueeze(0)
        parsing = helper.face_parse(TF.normalize(align, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0].argmax(dim=1, keepdim=True)
        background = sum(parsing == label for label in (0, 16, 18, 7, 8, 9, 14, 15)).bool()
        features = torch.where(background, torch.ones_like(align), to_gray(align))
        features = TF.normalize(TF.resize(features, eva.image_size, TF.InterpolationMode.BICUBIC), eva.image_mean, eva.image_std)
        id_cond_vit, id_vit_hidden = eva(features.to(device, eva_dtype), return_all_features=False, return_hidden=True, shuffle=False)
        id_cond_vit = id_cond_vit / torch.norm(id_cond_vit, 2, 1, True)
        id_cond = torch.cat([torch.from_numpy(face.embedding).unsqueeze(0).to(id_cond_vit), id_cond_vit], dim=-1)
        embeddings.append(models.pulid.model.pulid_encoder(id_cond, id_vit_hidden))
    if not embeddings:
        raise Exception("[Inspire Pack] Regional PuLID FLUX: facexlib found no face to align in the reference image.")
    return torch.cat(embeddings).mean(dim=0, keepdim=True).to(comfy.model_management.intermediate_device())


class PulidFluxApplication:
    def __init__(self, pulid, embedding, weight, sigma_range, mask):
        self.pulid = pulid
        self.embedding = embedding
        self.weight = weight
        self.sigma_range = sigma_range
        self.mask = mask

    def residual(self, ca_index, img, transformer_options):
        out = self.pulid.model.pulid_ca[ca_index](comfy.model_management.cast_to(self.embedding, img.dtype, img.device), img) * self.weight
        if self.mask is not None:
            out = out * token_mask(self.mask, transformer_options, img)
        return out

    def to(self, device_or_dtype):
        self.embedding = to_device(self.embedding, device_or_dtype)
        self.mask = to_device(self.mask, device_or_dtype)
        return self


class PulidFluxBlock:
    """patches_replace["dit"][(block, i)]: the block, or the patch it replaced, then the residuals of every PuLID FLUX
    application of this block, each from the same block output, added to the image tokens."""
    def __init__(self, ca_index, single, applications, previous):
        self.ca_index = ca_index
        self.single = single
        self.applications = applications
        self.previous = previous

    def __call__(self, args, extra_args):
        out = extra_args["original_block"](args) if self.previous is None else self.previous(args, extra_args)
        transformer_options = args["transformer_options"]
        active = [a for a in self.applications if in_range(a.sigma_range, transformer_options)]
        if active:
            img = out["img"][:, transformer_options["img_slice"][0]:] if self.single else out["img"]
            img += sum(a.residual(self.ca_index, img, transformer_options) for a in active)
        return out

    def to(self, device_or_dtype):
        for a in self.applications:
            a.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def models(self):
        return [a.pulid for a in self.applications] + (self.previous.models() if hasattr(self.previous, "models") else [])


def pulid_targets(diffusion_model):
    """The blocks after which PuLID FLUX adds its residual: every 2nd double block and every 4th single block of FLUX.1."""
    if not isinstance(diffusion_model, comfy.ldm.flux.model.Flux) or diffusion_model.hidden_size != 3072:
        return []
    return [("double_block", i) for i in range(0, len(diffusion_model.double_blocks), 2)] + [("single_block", i) for i in range(0, len(diffusion_model.single_blocks), 4)]


def apply_pulid_flux(model, models, embedding, weight, start_at, end_at, mask):
    targets = pulid_targets(model.get_model_object("diffusion_model"))
    if len(targets) != len(models.pulid.model.pulid_ca):
        raise Exception(f"[Inspire Pack] {models.file} is not a PuLID FLUX model for {base_name(model)}")

    model = model.clone()
    application = PulidFluxApplication(models.pulid, embedding, weight, sigma_range(model, start_at, end_at), mask_batch(mask))
    dit = model.model_options["transformer_options"].get("patches_replace", {}).get("dit", {})
    for ca_index, (kind, i) in enumerate(targets):
        previous = dit.get((kind, i))
        if isinstance(previous, PulidFluxBlock):
            patch = PulidFluxBlock(ca_index, kind == "single_block", previous.applications + [application], previous.previous)
        else:
            patch = PulidFluxBlock(ca_index, kind == "single_block", [application], previous)
        model.set_model_patch_replace(patch, "dit", kind, i)
    add_adapter_state(model)
    return model


class PulidFluxModelHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pulid_file": (folder_paths.get_filename_list("pulid"),),
                "provider": (["CPU", "CUDA", "ROCM"],),
                "cache_mode": (list(adapter_families["PuLID FLUX"]["cache_modes"]),),
            },
        }

    RETURN_TYPES = ("PULID_FLUX",)
    RETURN_NAMES = ("pulid_flux",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/models"

    @staticmethod
    def doit(pulid_file, provider, cache_mode):
        eva_clip, face_analysis = load_adapter_models("PulidFluxModelHelper", "PuLID FLUX", cache_mode, pulid_file=pulid_file, provider=provider)
        return (PulidFluxModels(pulid_file, eva_clip, face_analysis),)


class RegionalPulidFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pulid_flux": ("PULID_FLUX",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
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
    def doit(pulid_flux, image, weight, start_at, end_at, mask=None):
        require_face("PuLID FLUX", pulid_flux.face_analysis, image, 256)
        embedding = id_embedding(pulid_flux, image)

        def apply_model(model):
            return apply_pulid_flux(model, pulid_flux, embedding, weight, start_at, end_at, mask)

        return (RegionalAdapter(mask, apply_model=apply_model), )


NODE_CLASS_MAPPINGS = {
    "ReduxModelHelper //Inspire": ReduxModelHelper,
    "RegionalRedux //Inspire": RegionalRedux,
    "PulidFluxModelHelper //Inspire": PulidFluxModelHelper,
    "RegionalPulidFlux //Inspire": RegionalPulidFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReduxModelHelper //Inspire": "Redux Model Helper (Inspire)",
    "RegionalRedux //Inspire": "Regional Redux (Inspire)",
    "PulidFluxModelHelper //Inspire": "PuLID FLUX Model Helper (Inspire)",
    "RegionalPulidFlux //Inspire": "Regional PuLID FLUX (Inspire)",
}
