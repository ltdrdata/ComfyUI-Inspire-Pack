import nodes
import folder_paths
import os
import server
from .libs import utils

model_preset = {
    # base
    "SD1.5": ("ip-adapter_sd15", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 Light": ("ip-adapter_sd15_light", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 Plus": ("ip-adapter-plus_sd15", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 Plus Face": ("ip-adapter-plus-face_sd15", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 Full Face": ("ip-adapter-full-face_sd15", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SD1.5 ViT-G": ("ip-adapter_sd15_vit-G", "CLIP-ViT-bigG-14-laion2B-39B-b160k", None, False),
    "SDXL": ("ip-adapter_sdxl", "CLIP-ViT-bigG-14-laion2B-39B-b160k", None, False),
    "SDXL ViT-H": ("ip-adapter_sdxl_vit-h", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SDXL Plus ViT-H": ("ip-adapter-plus_sdxl_vit-h", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),
    "SDXL Plus Face ViT-H": ("ip-adapter-plus-face_sdxl_vit-h", "CLIP-ViT-H-14-laion2B-s32B-b79K", None, False),

    # faceid
    "SD1.5 FaceID": ("ip-adapter-faceid_sd15", None, "ip-adapter-faceid_sd15_lora", True),
    "SD1.5 FaceID Plus": ("ip-adapter-faceid-plus_sd15", "CLIP-ViT-H-14-laion2B-s32B-b79K", "ip-adapter-faceid-plus_sd15_lora", True),
    "SD1.5 FaceID Plus v2": ("ip-adapter-faceid-plusv2_sd15", "CLIP-ViT-H-14-laion2B-s32B-b79K", "ip-adapter-faceid-plusv2_sd15_lora", True),
    "SD1.5 FaceID Portrait": ("ip-adapter-faceid-portrait_sd15", None, None, True),
    "SDXL FaceID": ("ip-adapter-faceid_sdxl", None, "ip-adapter-faceid_sdxl_lora", True),
    "SDXL FaceID v2": ("ip-adapter-faceid-plusv2_sdxl", "CLIP-ViT-H-14-laion2B-s32B-b79K", "ip-adapter-faceid-plusv2_sdxl_lora", True),
    }


def lookup_model(model_dir, name):
    if name is None:
        return None, "N/A"

    names = [(os.path.splitext(os.path.basename(x))[0], x) for x in folder_paths.get_filename_list(model_dir)]
    resolved_name = [y for x, y in names if x == name]

    if len(resolved_name) > 0:
        return resolved_name[0], "OK"
    else:
        print(f"The `{name}` model file does not exist in `{model_dir}` model dir.")
        return None, "FAIL"


class IPAdapterModelHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "preset": (list(model_preset.keys()),),
                "lora_strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "lora_strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "insightface_provider": (["CPU", "CUDA", "ROCM"], ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IPADAPTER_PIPE", "MODEL", "CLIP", "IPADAPTER", "CLIP_VISION", "INSIGHTFACE")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/models"

    def doit(self, model, clip, preset, lora_strength_model, lora_strength_clip, insightface_provider, unique_id=None):
        if 'IPAdapterApply' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/cubiq/ComfyUI_IPAdapter_plus',
                                          "To use 'IPAdapterModelHelper' node, 'ComfyUI IPAdapter Plus' extension is required.")
            raise Exception(f"[ERROR] To use IPAdapterModelHelper, you need to install 'ComfyUI IPAdapter Plus'")

        ipadapter, clipvision, lora, is_insightface = model_preset[preset]

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

        server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 3, "label": ok1})
        server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 4, "label": ok2})

        if ok3 == "FAIL":
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 1, "label": "MODEL (fail)"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 2, "label": "CLIP (fail)"})
        else:
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 1, "label": "MODEL"})
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 2, "label": "CLIP"})

        if ok1 == "FAIL" or ok2 == "FAIL" and ok3 == "FAIL":
            raise Exception("ERROR: Failed to load several models in IPAdapterModelHelper.")

        if ipadapter is not None:
            ipadapter = nodes.NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]().load_ipadapter_model(ipadapter_file=ipadapter)[0]

        if clipvision is not None:
            clipvision = nodes.CLIPVisionLoader().load_clip(clip_name=clipvision)[0]

        if lora is not None:
            model, clip = nodes.LoraLoader().load_lora(model=model, clip=clip, lora_name=lora, strength_model=lora_strength_model, strength_clip=lora_strength_clip)[0]

        if is_insightface:
            insightface = nodes.NODE_CLASS_MAPPINGS["InsightFaceLoader"]().load_insight_face(insightface_provider)[0]
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 5, "label": "INSIGHTFACE"})
        else:
            insightface = None
            server.PromptServer.instance.send_sync("inspire-node-output-label", {"node_id": unique_id, "output_idx": 5, "label": "INSIGHTFACE (N/A)"})

        pipe = ipadapter, clipvision, model
        return pipe, model, clip, ipadapter, clipvision, insightface


NODE_CLASS_MAPPINGS = {
    "IPAdapterModelHelper //Inspire": IPAdapterModelHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterModelHelper //Inspire": "IPAdapter Model Helper (Inspire)",
}
