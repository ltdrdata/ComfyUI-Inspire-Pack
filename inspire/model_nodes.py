import nodes
import folder_paths
import os
import safetensors
import server
from .libs import utils
from . import backend_support
from comfy import sdxl_clip
import logging


utils.add_folder_path_and_extensions("ipadapter", [os.path.join(folder_paths.models_dir, "ipadapter")], folder_paths.supported_pt_extensions)


model_preset = {
    # base
    "SD1.5":                ("ip-adapter_sd15",                 "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 Light v11":      ("ip-adapter_sd15_light_v11",       "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 Light":          ("ip-adapter_sd15_light",           "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 Plus":           ("ip-adapter-plus_sd15",            "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 Plus Face":      ("ip-adapter-plus-face_sd15",       "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 Full Face":      ("ip-adapter-full-face_sd15",       "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 ViT-G":          ("ip-adapter_sd15_vit-G",           "CLIP-ViT-bigG-14-laion2B-39B-b160k", None, False),
    "SDXL":                 ("ip-adapter_sdxl",                 "CLIP-ViT-bigG-14-laion2B-39B-b160k", None, False),
    "SDXL ViT-H":           ("ip-adapter_sdxl_vit-h",           "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SDXL Plus ViT-H":      ("ip-adapter-plus_sdxl_vit-h",      "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SDXL Plus Face ViT-H": ("ip-adapter-plus-face_sdxl_vit-h", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "Kolors Plus":          ("Kolors-IP-Adapter-Plus",          "clip-vit-large-patch14-336", None, False),

    # faceid
    "SD1.5 FaceID":                ("ip-adapter-faceid_sd15",                 "CLIP-ViT-H-14-laion2B-s32B-b79K", "ip-adapter-faceid_sd15_lora", True),
    "SD1.5 FaceID Plus v2":        ("ip-adapter-faceid-plusv2_sd15",          "CLIP-ViT-H-14-laion2B-s32B-b79K", "ip-adapter-faceid-plusv2_sd15_lora", True),
    "SD1.5 FaceID Plus":           ("ip-adapter-faceid-plus_sd15",            "CLIP-ViT-H-14-laion2B-s32B-b79K", "ip-adapter-faceid-plus_sd15_lora", True),
    "SD1.5 FaceID Portrait v11":   ("ip-adapter-faceid-portrait-v11_sd15",    "CLIP-ViT-H-14-laion2B-s32B-b79K", None, True),
    "SD1.5 FaceID Portrait":       ("ip-adapter-faceid-portrait_sd15",        "CLIP-ViT-H-14-laion2B-s32B-b79K", None, True),
    "SDXL FaceID":                 ("ip-adapter-faceid_sdxl",                 "CLIP-ViT-H-14-laion2B-s32B-b79K", "ip-adapter-faceid_sdxl_lora", True),
    "SDXL FaceID Plus v2":         ("ip-adapter-faceid-plusv2_sdxl",          "CLIP-ViT-H-14-laion2B-s32B-b79K", "ip-adapter-faceid-plusv2_sdxl_lora", True),
    "SDXL FaceID Portrait":        ("ip-adapter-faceid-portrait_sdxl",        "CLIP-ViT-H-14-laion2B-s32B-b79K", None, True),
    "SDXL FaceID Portrait unnorm": ("ip-adapter-faceid-portrait_sdxl_unnorm", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, True),
    "Kolors FaceID Plus":          ("Kolors-IP-Adapter-FaceID-Plus",          "clip-vit-large-patch14-336", None, True),

    # composition
    "SD1.5 Plus Composition":      ("ip_plus_composition_sd15", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SDXL Plus Composition":       ("ip_plus_composition_sdxl", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    }


def lookup_model(model_dir, name):
    if name is None:
        return None, "N/A"

    names = [(os.path.splitext(os.path.basename(x))[0], x) for x in folder_paths.get_filename_list(model_dir)]
    resolved_name = [y for x, y in names if x == name]

    if len(resolved_name) > 0:
        return resolved_name[0], "OK"
    else:
        logging.error(f"[Inspire Pack] IPAdapterModelHelper: The `{name}` model file does not exist in `{model_dir}` model dir.")
        return None, "FAIL"


class IPAdapterModelHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (list(model_preset.keys()),),
                "lora_strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "lora_strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "insightface_provider": (["CPU", "CUDA", "ROCM"], ),
                "cache_mode": (["insightface only", "clip_vision only", "all", "none"], {"default": "insightface only"}),
            },
            "optional": {
                "clip": ("CLIP",),
                "insightface_model_name": (['buffalo_l', 'antelopev2'],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IPADAPTER_PIPE", "IPADAPTER", "CLIP_VISION", "INSIGHTFACE", "MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("IPADAPTER_PIPE", "IPADAPTER", "CLIP_VISION", "INSIGHTFACE", "MODEL", "CLIP", "insightface_cache_key", "clip_vision_cache_key")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/models"

    def doit(self, model, preset, lora_strength_model, lora_strength_clip, insightface_provider, clip=None, cache_mode="none", unique_id=None, insightface_model_name='buffalo_l'):
        if 'IPAdapter' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/cubiq/ComfyUI_IPAdapter_plus',
                                          "To use 'IPAdapterModelHelper' node, 'ComfyUI IPAdapter Plus' extension is required.")
            raise Exception("[ERROR] To use IPAdapterModelHelper, you need to install 'ComfyUI IPAdapter Plus'")

        is_sdxl_preset = 'SDXL' in preset
        if clip is not None:
            is_sdxl_model = isinstance(clip.tokenizer, sdxl_clip.SDXLTokenizer)
        else:
            is_sdxl_model = False

        if is_sdxl_preset != is_sdxl_model:
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 1, "label": "IPADAPTER (fail)"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 2, "label": "CLIP_VISION (fail)"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 3, "label": "INSIGHTFACE (fail)"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 4, "label": "MODEL (fail)"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 5, "label": "CLIP (fail)"})
            logging.error("[Inspire Pack] IPAdapterModelHelper: You cannot mix SDXL and SD1.5 in the checkpoint and IPAdapter.")
            raise Exception("[ERROR] You cannot mix SDXL and SD1.5 in the checkpoint and IPAdapter.")

        ipadapter, clipvision, lora, is_insightface = model_preset[preset]

        ipadapter, ok1 = lookup_model("ipadapter", ipadapter)
        clipvision, ok2 = lookup_model("clip_vision", clipvision)
        lora, ok3 = lookup_model("loras", lora)

        server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 1, "label": "IPADAPTER" if ok1 == "OK" else f"IPADAPTER ({ok1})"})
        server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 2, "label": "CLIP_VISION" if ok2 == "OK" else f"CLIP_VISION ({ok2})"})

        if ok3 == "FAIL":
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 4, "label": "MODEL (fail)"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 5, "label": "CLIP (fail)"})
        else:
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 4, "label": "MODEL"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 5, "label": "CLIP"})

        if ok1 == "FAIL" or ok2 == "FAIL" or ok3 == "FAIL":
            raise Exception("ERROR: Failed to load several models in IPAdapterModelHelper.")

        return self.load_models(model, clip, ipadapter, clipvision, lora, is_insightface, lora_strength_model, lora_strength_clip, insightface_provider, cache_mode, unique_id, insightface_model_name)

    @staticmethod
    def load_models(model, clip, ipadapter, clipvision, lora, is_insightface, lora_strength_model, lora_strength_clip, insightface_provider, cache_mode, unique_id, insightface_model_name):
        if ipadapter is not None:
            ipadapter = nodes.NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]().load_ipadapter_model(ipadapter_file=ipadapter)[0]

        ccache_key = ""
        if clipvision is not None:
            if cache_mode in ["clip_vision only", "all"]:
                ccache_key = clipvision
                if ccache_key not in backend_support.cache:
                    backend_support.update_cache(ccache_key, "clipvision", (False, nodes.CLIPVisionLoader().load_clip(clip_name=clipvision)[0]))
                _, (_, clipvision) = backend_support.cache[ccache_key]
            else:
                clipvision = nodes.CLIPVisionLoader().load_clip(clip_name=clipvision)[0]

        if lora is not None:
            model, clip = nodes.LoraLoader().load_lora(model=model, clip=clip, lora_name=lora, strength_model=lora_strength_model, strength_clip=lora_strength_clip)

            def f(x):
                return nodes.LoraLoader().load_lora(model=x, clip=clip, lora_name=lora, strength_model=lora_strength_model, strength_clip=lora_strength_clip)
            lora_loader = f
        else:
            def f(x):
                return x
            lora_loader = f

        if 'IPAdapterInsightFaceLoader' in nodes.NODE_CLASS_MAPPINGS:
            insight_face_loader = nodes.NODE_CLASS_MAPPINGS['IPAdapterInsightFaceLoader']().load_insightface
        else:
            logging.warning("'ComfyUI IPAdapter Plus' extension is either too outdated or not installed.")
            insight_face_loader = None

        icache_key = ""
        if is_insightface:
            if insight_face_loader is None:
                raise Exception("[ERROR] 'ComfyUI IPAdapter Plus' extension is either too outdated or not installed.")

            if cache_mode in ["insightface only", "all"]:
                icache_key = 'insightface-' + insightface_model_name + '-' + insightface_provider
                if icache_key not in backend_support.cache:
                    backend_support.update_cache(icache_key, "insightface", (False, insight_face_loader(provider=insightface_provider, model_name=insightface_model_name)[0]))
                _, (_, insightface) = backend_support.cache[icache_key]
            else:
                insightface = insight_face_loader(provider=insightface_provider, model_name=insightface_model_name)[0]

            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 3, "label": "INSIGHTFACE"})
        else:
            insightface = None
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 3, "label": "INSIGHTFACE (N/A)"})

        pipe = ipadapter, model, clipvision, insightface, lora_loader
        return pipe, ipadapter, clipvision, insightface, model, clip, icache_key, ccache_key


class IPAdapterModelHelperManual(IPAdapterModelHelper):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter_model": (folder_paths.get_filename_list("ipadapter"),),
                "clip_vision_model": (folder_paths.get_filename_list("clip_vision"),),
                "lora_model": (["None"] + folder_paths.get_filename_list("loras"),),
                "is_insightface": ("BOOLEAN", {"default": False}),
                "lora_strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "lora_strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "insightface_provider": (["CPU", "CUDA", "ROCM"], ),
                "cache_mode": (["insightface only", "clip_vision only", "all", "none"], {"default": "insightface only"}),
            },
            "optional": {
                "clip": ("CLIP",),
                "insightface_model_name": (['buffalo_l', 'antelopev2'],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    def doit(self, model, ipadapter_model, clip_vision_model, lora_model, is_insightface, lora_strength_model, lora_strength_clip, insightface_provider, clip=None, cache_mode="none", unique_id=None, insightface_model_name='buffalo_l'):
        if 'IPAdapter' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/cubiq/ComfyUI_IPAdapter_plus',
                                          "To use 'IPAdapterModelHelper' node, 'ComfyUI IPAdapter Plus' extension is required.")
            raise Exception("[ERROR] To use IPAdapterModelHelper, you need to install 'ComfyUI IPAdapter Plus'")

        lora = None if lora_model == "None" else lora_model
        return self.load_models(model, clip, ipadapter_model, clip_vision_model, lora, is_insightface, lora_strength_model, lora_strength_clip, insightface_provider, cache_mode, unique_id, insightface_model_name)


# The node packs of the adapter families: the pack to ask for when it is missing, the model folder it reads, and its
# loaders in the order of the helper's outputs as (node, function, inputs, cache name, cache key). cache_modes maps each
# choice of the helper's cache_mode to the cache names it keeps in the backend cache. signatures are (input, key prefix,
# base model): a .safetensors file picked in that input must have a key with that prefix.
adapter_families = {
    "PuLID": {
        "pack": ("PuLID_ComfyUI", "https://github.com/cubiq/PuLID_ComfyUI"),
        "folder": "pulid",
        "loaders": [("PulidModelLoader", "load_model", ["pulid_file"], None, None),
                    ("PulidEvaClipLoader", "load_eva_clip", [], "eva_clip", "pulid-eva_clip"),
                    ("PulidInsightFaceLoader", "load_insightface", ["provider"], "insightface", "insightface-antelopev2-{provider}")],
        "cache_modes": {"insightface only": ["insightface"], "eva_clip only": ["eva_clip"], "all": ["insightface", "eva_clip"], "none": []},
        "signatures": [("pulid_file", "image_proj.mapping_", "SDXL")],
    },
    "InstantID": {
        "pack": ("ComfyUI InstantID", "https://github.com/cubiq/ComfyUI_InstantID"),
        "folder": "instantid",
        "loaders": [("InstantIDModelLoader", "load_model", ["instantid_file"], None, None),
                    ("InstantIDFaceAnalysis", "load_insight_face", ["provider"], "insightface", "insightface-antelopev2-{provider}"),
                    ("ControlNetLoader", "load_controlnet", ["control_net_name"], None, None)],
        "cache_modes": {"insightface only": ["insightface"], "none": []},
        "signatures": [],
    },
}

for family in adapter_families.values():
    utils.add_folder_path_and_extensions(family["folder"], [os.path.join(folder_paths.models_dir, family["folder"])], folder_paths.supported_pt_extensions)


def load_adapter_models(node_name, family_name, cache_mode, **inputs):
    family = adapter_families[family_name]
    pack, url = family["pack"]
    if any(loader not in nodes.NODE_CLASS_MAPPINGS for loader, *_ in family["loaders"]):
        utils.try_install_custom_node(url, f"To use '{node_name}' node, '{pack}' extension is required.")
        raise Exception(f"[ERROR] To use {node_name}, you need to install '{pack}'")

    for name, prefix, base in family["signatures"]:
        path = folder_paths.get_full_path_or_raise(family["folder"], inputs[name])
        if path.endswith(".safetensors"):
            with safetensors.safe_open(path, framework="pt") as f:
                if not any(key.startswith(prefix) for key in f.keys()):
                    raise Exception(f"[Inspire Pack] {inputs[name]} is not a {family_name} model for {base}")

    models = []
    for loader, function, names, cache_name, cache_key in family["loaders"]:
        def load():
            return getattr(nodes.NODE_CLASS_MAPPINGS[loader](), function)(**{name: inputs[name] for name in names})[0]

        if cache_name in family["cache_modes"][cache_mode]:
            key = cache_key.format(**inputs)
            if key not in backend_support.cache:
                backend_support.update_cache(key, cache_name, (False, load()))
            _, (_, model) = backend_support.cache[key]
        else:
            model = load()
        models.append(model)

    return tuple(models)


class PulidModelHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pulid_file": (folder_paths.get_filename_list("pulid"),),
                "provider": (["CPU", "CUDA", "ROCM"],),
                "cache_mode": (list(adapter_families["PuLID"]["cache_modes"]),),
            },
        }

    RETURN_TYPES = ("PULID", "EVA_CLIP", "FACEANALYSIS")
    RETURN_NAMES = ("pulid", "eva_clip", "face_analysis")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/models"

    @staticmethod
    def doit(pulid_file, provider, cache_mode):
        return load_adapter_models("PulidModelHelper", "PuLID", cache_mode, pulid_file=pulid_file, provider=provider)


class InstantIDModelHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid_file": (folder_paths.get_filename_list("instantid"),),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "provider": (["CPU", "CUDA", "ROCM", "CoreML"],),
                "cache_mode": (list(adapter_families["InstantID"]["cache_modes"]),),
            },
        }

    RETURN_TYPES = ("INSTANTID", "FACEANALYSIS", "CONTROL_NET")
    RETURN_NAMES = ("instantid", "insightface", "control_net")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/models"

    @staticmethod
    def doit(instantid_file, control_net_name, provider, cache_mode):
        return load_adapter_models("InstantIDModelHelper", "InstantID", cache_mode,
                                   instantid_file=instantid_file, control_net_name=control_net_name, provider=provider)


NODE_CLASS_MAPPINGS = {
    "IPAdapterModelHelper //Inspire": IPAdapterModelHelper,
    "IPAdapterModelHelperManual //Inspire": IPAdapterModelHelperManual,
    "PulidModelHelper //Inspire": PulidModelHelper,
    "InstantIDModelHelper //Inspire": InstantIDModelHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterModelHelper //Inspire": "IPAdapter Model Helper (Inspire)",
    "IPAdapterModelHelperManual //Inspire": "IPAdapter Model Helper [Manual] (Inspire)",
    "PulidModelHelper //Inspire": "PuLID Model Helper (Inspire)",
    "InstantIDModelHelper //Inspire": "InstantID Model Helper (Inspire)",
}
