import nodes
import folder_paths
import os
import server
from .libs import utils
from . import backend_support
from comfy import sdxl_clip
import logging


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
        # Scan for IPAdapter model files in the ipadapter folders
        ipadapter_options = []
        try:
            ipadapter_files = folder_paths.get_filename_list("ipadapter")
            for filename in ipadapter_files:
                # Remove extension to get the base name
                base_name = os.path.splitext(filename)[0]
                ipadapter_options.append(base_name)
        except Exception as e:
            logging.warning(f"[Inspire Pack] IPAdapterModelHelper: Failed to scan ipadapter folder: {e}")
        
        # Scan for CLIP vision model files in the clip_vision folders
        clipvision_options = []
        try:
            clip_vision_files = folder_paths.get_filename_list("clip_vision")
            for filename in clip_vision_files:
                # Remove extension to get the base name
                base_name = os.path.splitext(filename)[0]
                clipvision_options.append(base_name)
        except Exception as e:
            logging.warning(f"[Inspire Pack] IPAdapterModelHelper: Failed to scan clip_vision folder: {e}")
        
        # Scan for LoRA model files in the loras folders
        lora_options = ["None"]  # Add "None" as first option for no LoRA
        try:
            lora_files = folder_paths.get_filename_list("loras")
            for filename in lora_files:
                # Remove extension to get the base name
                base_name = os.path.splitext(filename)[0]
                lora_options.append(base_name)
        except Exception as e:
            logging.warning(f"[Inspire Pack] IPAdapterModelHelper: Failed to scan loras folder: {e}")
        
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter_model": (ipadapter_options,),
                "clip_vision_model": (clipvision_options,),
                "lora_model": (lora_options,),
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

    RETURN_TYPES = ("IPADAPTER_PIPE", "IPADAPTER", "CLIP_VISION", "INSIGHTFACE", "MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("IPADAPTER_PIPE", "IPADAPTER", "CLIP_VISION", "INSIGHTFACE", "MODEL", "CLIP", "insightface_cache_key", "clip_vision_cache_key")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/models"

    def doit(self, model, ipadapter_model, clip_vision_model, lora_model, is_insightface, lora_strength_model, lora_strength_clip, insightface_provider, clip=None, cache_mode="none", unique_id=None, insightface_model_name='buffalo_l'):
        if 'IPAdapter' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/cubiq/ComfyUI_IPAdapter_plus',
                                          "To use 'IPAdapterModelHelper' node, 'ComfyUI IPAdapter Plus' extension is required.")
            raise Exception("[ERROR] To use IPAdapterModelHelper, you need to install 'ComfyUI IPAdapter Plus'")

        # Use the selected models directly
        ipadapter = ipadapter_model
        clipvision = clip_vision_model
        lora = None if lora_model == "None" else lora_model

        ipadapter, ok1 = lookup_model("ipadapter", ipadapter)
        clipvision, ok2 = lookup_model("clip_vision", clipvision)
        lora, ok3 = lookup_model("loras", lora)

        if ok1 == "OK":
            ok1 = "IPADAPTER"
        else:
            ok1 = f"IPADAPTER ({ok1})"

        if ok2 == "OK":
            ok2 = "CLIP_VISION"
        else:
            ok2 = f"CLIP_VISION ({ok2})"

        server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 1, "label": ok1})
        server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 2, "label": ok2})

        if ok3 == "FAIL":
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 4, "label": "MODEL (fail)"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 5, "label": "CLIP (fail)"})
        else:
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 4, "label": "MODEL"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 5, "label": "CLIP"})

        if ok1 == "FAIL" or ok2 == "FAIL" or ok3 == "FAIL":
            raise Exception("ERROR: Failed to load several models in IPAdapterModelHelper.")

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
                icache_key = 'insightface-' + insightface_provider
                if icache_key not in backend_support.cache:
                    backend_support.update_cache(icache_key, "insightface", (False, insight_face_loader(provider=insightface_provider, model_name=insightface_model_name)[0]))
                _, (_, insightface) = backend_support.cache[icache_key]
            else:
                insightface = insight_face_loader(insightface_provider)[0]

            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 3, "label": "INSIGHTFACE"})
        else:
            insightface = None
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 3, "label": "INSIGHTFACE (N/A)"})

        pipe = ipadapter, model, clipvision, insightface, lora_loader
        return pipe, ipadapter, clipvision, insightface, model, clip, icache_key, ccache_key


NODE_CLASS_MAPPINGS = {
    "IPAdapterModelHelper //Inspire": IPAdapterModelHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterModelHelper //Inspire": "IPAdapter Model Helper (Inspire)",
}
