import nodes
import numpy as np
import torch
from .libs import utils
import logging

def normalize_size_base_64(w, h):
    short_side = min(w, h)
    remainder = short_side % 64
    return short_side - remainder + (64 if remainder > 0 else 0)


class MediaPipeFaceMeshDetector:
    def __init__(self, face, mouth, left_eyebrow, left_eye, left_pupil, right_eyebrow, right_eye, right_pupil, max_faces, is_segm):
        self.face = face
        self.mouth = mouth
        self.left_eyebrow = left_eyebrow
        self.left_eye = left_eye
        self.left_pupil = left_pupil
        self.right_eyebrow = right_eyebrow
        self.right_eye = right_eye
        self.right_pupil = right_pupil
        self.is_segm = is_segm
        self.max_faces = max_faces
        self.override_bbox_by_segm = True

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, crop_min_size=None, detailer_hook=None):
        if 'MediaPipe-FaceMeshPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'MediaPipeFaceMeshDetector' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use MediaPipeFaceMeshDetector, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        if 'MediaPipeFaceMeshToSEGS' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Impact-Pack',
                                          "To use 'MediaPipeFaceMeshDetector' node, 'Impact Pack' extension is required.")
            raise Exception("[ERROR] To use MediaPipeFaceMeshDetector, you need to install 'ComfyUI-Impact-Pack'")

        pre_obj = nodes.NODE_CLASS_MAPPINGS['MediaPipe-FaceMeshPreprocessor']
        seg_obj = nodes.NODE_CLASS_MAPPINGS['MediaPipeFaceMeshToSEGS']

        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        facemesh_image = pre_obj().detect(image, self.max_faces, threshold, resolution=resolution)[0]

        facemesh_image = nodes.ImageScale().upscale(facemesh_image, "bilinear", image.shape[2], image.shape[1], "disabled")[0]

        segs = seg_obj().doit(facemesh_image, crop_factor, not self.is_segm, crop_min_size, drop_size, dilation,
                              self.face, self.mouth, self.left_eyebrow, self.left_eye, self.left_pupil,
                              self.right_eyebrow, self.right_eye, self.right_pupil)[0]

        return segs

    def setAux(self, x):
        pass


class MediaPipe_FaceMesh_Preprocessor_wrapper:
    def __init__(self, max_faces, min_confidence, upscale_factor=1.0):
        self.max_faces = max_faces
        self.min_confidence = min_confidence
        self.upscale_factor = upscale_factor

    def apply(self, image, mask=None):
        if 'MediaPipe-FaceMeshPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'MediaPipe_FaceMesh_Preprocessor_Provider_for_SEGS' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use MediaPipe_FaceMesh_Preprocessor_Provider_for_SEGS, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        if self.upscale_factor != 1.0:
            image = nodes.ImageScaleBy().upscale(image, 'bilinear', self.upscale_factor)[0]

        obj = nodes.NODE_CLASS_MAPPINGS['MediaPipe-FaceMeshPreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.detect(image, self.max_faces, self.min_confidence, resolution=resolution)[0]


class AnimeLineArt_Preprocessor_wrapper:
    def apply(self, image, mask=None):
        if 'AnimeLineArtPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'AnimeLineArt_Preprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use AnimeLineArt_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['AnimeLineArtPreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, resolution=resolution)[0]


class Manga2Anime_LineArt_Preprocessor_wrapper:
    def apply(self, image, mask=None):
        if 'Manga2Anime_LineArt_Preprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'Manga2Anime_LineArt_Preprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use Manga2Anime_LineArt_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['Manga2Anime_LineArt_Preprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, resolution=resolution)[0]


class Color_Preprocessor_wrapper:
    def apply(self, image, mask=None):
        if 'ColorPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'Color_Preprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use Color_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['ColorPreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, resolution=resolution)[0]


class InpaintPreprocessor_wrapper:
    def __init__(self, black_pixel_for_xinsir_cn):
        self.black_pixel_for_xinsir_cn = black_pixel_for_xinsir_cn

    def apply(self, image, mask=None):
        if 'InpaintPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'InpaintPreprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use InpaintPreprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['InpaintPreprocessor']()
        if mask is None:
            mask = torch.ones((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu").unsqueeze(0)

        try:
            res = obj.preprocess(image, mask, black_pixel_for_xinsir_cn=self.black_pixel_for_xinsir_cn)[0]
        except Exception as e:
            if self.black_pixel_for_xinsir_cn:
                raise e
            else:
                res = obj.preprocess(image, mask)[0]
                logging.warning("[Inspire Pack] Installed 'ComfyUI's ControlNet Auxiliary Preprocessors.' is outdated.")

        return res


class TilePreprocessor_wrapper:
    def __init__(self, pyrUp_iters):
        self.pyrUp_iters = pyrUp_iters

    def apply(self, image, mask=None):
        if 'TilePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'TilePreprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use TilePreprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['TilePreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, self.pyrUp_iters, resolution=resolution)[0]


class MeshGraphormerDepthMapPreprocessorProvider_wrapper:
    def apply(self, image, mask=None):
        if 'MeshGraphormer-DepthMapPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'MeshGraphormerDepthMapPreprocessorProvider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use MeshGraphormerDepthMapPreprocessorProvider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['MeshGraphormer-DepthMapPreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, resolution=resolution)[0]


class LineArt_Preprocessor_wrapper:
    def __init__(self, coarse):
        self.coarse = coarse

    def apply(self, image, mask=None):
        if 'LineArtPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'LineArt_Preprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use LineArt_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        coarse = 'enable' if self.coarse else 'disable'

        obj = nodes.NODE_CLASS_MAPPINGS['LineArtPreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, resolution=resolution, coarse=coarse)[0]


class OpenPose_Preprocessor_wrapper:
    def __init__(self, detect_hand, detect_body, detect_face, upscale_factor=1.0):
        self.detect_hand = detect_hand
        self.detect_body = detect_body
        self.detect_face = detect_face
        self.upscale_factor = upscale_factor

    def apply(self, image, mask=None):
        if 'OpenposePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'OpenPose_Preprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use OpenPose_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        detect_hand = 'enable' if self.detect_hand else 'disable'
        detect_body = 'enable' if self.detect_body else 'disable'
        detect_face = 'enable' if self.detect_face else 'disable'

        if self.upscale_factor != 1.0:
            image = nodes.ImageScaleBy().upscale(image, 'bilinear', self.upscale_factor)[0]

        obj = nodes.NODE_CLASS_MAPPINGS['OpenposePreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.estimate_pose(image, detect_hand, detect_body, detect_face, resolution=resolution)['result'][0]


class DWPreprocessor_wrapper:
    def __init__(self, detect_hand, detect_body, detect_face, upscale_factor=1.0, bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx"):
        self.detect_hand = detect_hand
        self.detect_body = detect_body
        self.detect_face = detect_face
        self.upscale_factor = upscale_factor
        self.bbox_detector = bbox_detector
        self.pose_estimator = pose_estimator

    def apply(self, image, mask=None):
        if 'DWPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'DWPreprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use DWPreprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        detect_hand = 'enable' if self.detect_hand else 'disable'
        detect_body = 'enable' if self.detect_body else 'disable'
        detect_face = 'enable' if self.detect_face else 'disable'

        if self.upscale_factor != 1.0:
            image = nodes.ImageScaleBy().upscale(image, 'bilinear', self.upscale_factor)[0]

        obj = nodes.NODE_CLASS_MAPPINGS['DWPreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.estimate_pose(image, detect_hand, detect_body, detect_face, resolution=resolution, bbox_detector=self.bbox_detector, pose_estimator=self.pose_estimator)['result'][0]


class LeReS_DepthMap_Preprocessor_wrapper:
    def __init__(self, rm_nearest, rm_background, boost):
        self.rm_nearest = rm_nearest
        self.rm_background = rm_background
        self.boost = boost

    def apply(self, image, mask=None):
        if 'LeReS-DepthMapPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'LeReS_DepthMap_Preprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use LeReS_DepthMap_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        boost = 'enable' if self.boost else 'disable'

        obj = nodes.NODE_CLASS_MAPPINGS['LeReS-DepthMapPreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, self.rm_nearest, self.rm_background, boost=boost, resolution=resolution)[0]


class MiDaS_DepthMap_Preprocessor_wrapper:
    def __init__(self, a, bg_threshold):
        self.a = a
        self.bg_threshold = bg_threshold

    def apply(self, image, mask=None):
        if 'MiDaS-DepthMapPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'MiDaS_DepthMap_Preprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use MiDaS_DepthMap_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['MiDaS-DepthMapPreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, self.a, self.bg_threshold, resolution=resolution)[0]


class Zoe_DepthMap_Preprocessor_wrapper:
    def apply(self, image, mask=None):
        if 'Zoe-DepthMapPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          "To use 'Zoe_DepthMap_Preprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception("[ERROR] To use Zoe_DepthMap_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['Zoe-DepthMapPreprocessor']()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, resolution=resolution)[0]


class HED_Preprocessor_wrapper:
    def __init__(self, safe, nodename):
        self.safe = safe
        self.nodename = nodename

    def apply(self, image, mask=None):
        if self.nodename not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/Fannovel16/comfyui_controlnet_aux',
                                          f"To use '{self.nodename}_Preprocessor_Provider' node, 'ComfyUI's ControlNet Auxiliary Preprocessors.' extension is required.")
            raise Exception(f"[ERROR] To use {self.nodename}_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS[self.nodename]()
        resolution = normalize_size_base_64(image.shape[2], image.shape[1])
        return obj.execute(image, resolution=resolution, safe="enable" if self.safe else "disable")[0]


class Canny_Preprocessor_wrapper:
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def apply(self, image, mask=None):
        obj = nodes.NODE_CLASS_MAPPINGS['Canny']()
        if hasattr(obj, 'execute'):
            # node v3
            return obj.execute(image, self.low_threshold, self.high_threshold)[0]
        else:
            # legacy compatibility
            return obj.detect_edge(image, self.low_threshold, self.high_threshold)[0]


class OpenPose_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detect_hand": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "detect_body": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "detect_face": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "resolution_upscale_by": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 100, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, detect_hand, detect_body, detect_face, resolution_upscale_by):
        obj = OpenPose_Preprocessor_wrapper(detect_hand, detect_body, detect_face, upscale_factor=resolution_upscale_by)
        return (obj, )


class DWPreprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detect_hand": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "detect_body": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "detect_face": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "resolution_upscale_by": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 100, "step": 0.1}),
                "bbox_detector": (
                    ["yolox_l.torchscript.pt", "yolox_l.onnx", "yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx"],
                    {"default": "yolox_l.onnx"}
                ),
                "pose_estimator": (["dw-ll_ucoco_384_bs5.torchscript.pt", "dw-ll_ucoco_384.onnx", "dw-ll_ucoco.onnx"], {"default": "dw-ll_ucoco_384_bs5.torchscript.pt"})
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, detect_hand, detect_body, detect_face, resolution_upscale_by, bbox_detector, pose_estimator):
        obj = DWPreprocessor_wrapper(detect_hand, detect_body, detect_face, upscale_factor=resolution_upscale_by, bbox_detector=bbox_detector, pose_estimator=pose_estimator)
        return (obj, )


class LeReS_DepthMap_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rm_nearest": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "rm_background": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1})
            },
            "optional": {
                "boost": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, rm_nearest, rm_background, boost=False):
        obj = LeReS_DepthMap_Preprocessor_wrapper(rm_nearest, rm_background, boost)
        return (obj, )


class MiDaS_DepthMap_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 5.0, "step": 0.05}),
                "bg_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.05})
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, a, bg_threshold):
        obj = MiDaS_DepthMap_Preprocessor_wrapper(a, bg_threshold)
        return (obj, )


class Zoe_DepthMap_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self):
        obj = Zoe_DepthMap_Preprocessor_wrapper()
        return (obj, )


class Canny_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "low_threshold": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 0.99, "step": 0.01}),
                "high_threshold": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 0.99, "step": 0.01})
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, low_threshold, high_threshold):
        obj = Canny_Preprocessor_wrapper(low_threshold, high_threshold)
        return (obj, )


class HEDPreprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "safe": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"})
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, safe):
        obj = HED_Preprocessor_wrapper(safe, "HEDPreprocessor")
        return (obj, )


class FakeScribblePreprocessor_Provider_for_SEGS(HEDPreprocessor_Provider_for_SEGS):
    def doit(self, safe):
        obj = HED_Preprocessor_wrapper(safe, "FakeScribblePreprocessor")
        return (obj, )


class MediaPipe_FaceMesh_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_faces": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "resolution_upscale_by": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 100, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, max_faces, min_confidence, resolution_upscale_by):
        obj = MediaPipe_FaceMesh_Preprocessor_wrapper(max_faces, min_confidence, upscale_factor=resolution_upscale_by)
        return (obj, )


class MediaPipeFaceMeshDetectorProvider:
    @classmethod
    def INPUT_TYPES(s):
        bool_true_widget = ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"})
        bool_false_widget = ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"})
        return {"required": {
                                "max_faces": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                                "face": bool_true_widget,
                                "mouth": bool_false_widget,
                                "left_eyebrow": bool_false_widget,
                                "left_eye": bool_false_widget,
                                "left_pupil": bool_false_widget,
                                "right_eyebrow": bool_false_widget,
                                "right_eye": bool_false_widget,
                                "right_pupil": bool_false_widget,
                            }}

    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Detector"

    def doit(self, max_faces, face, mouth, left_eyebrow, left_eye, left_pupil, right_eyebrow, right_eye, right_pupil):
        bbox_detector = MediaPipeFaceMeshDetector(face, mouth, left_eyebrow, left_eye, left_pupil, right_eyebrow, right_eye, right_pupil, max_faces, is_segm=False)
        segm_detector = MediaPipeFaceMeshDetector(face, mouth, left_eyebrow, left_eye, left_pupil, right_eyebrow, right_eye, right_pupil, max_faces, is_segm=True)

        return (bbox_detector, segm_detector)


class AnimeLineArt_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self):
        obj = AnimeLineArt_Preprocessor_wrapper()
        return (obj, )


class Manga2Anime_LineArt_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self):
        obj = Manga2Anime_LineArt_Preprocessor_wrapper()
        return (obj, )


class LineArt_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "coarse": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
        }}
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, coarse):
        obj = LineArt_Preprocessor_wrapper(coarse)
        return (obj, )


class Color_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self):
        obj = Color_Preprocessor_wrapper()
        return (obj, )


class InpaintPreprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "black_pixel_for_xinsir_cn": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, black_pixel_for_xinsir_cn=False):
        obj = InpaintPreprocessor_wrapper(black_pixel_for_xinsir_cn)
        return (obj, )


class TilePreprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {'pyrUp_iters': ("INT", {"default": 3, "min": 1, "max": 10, "step": 1})}}
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, pyrUp_iters):
        obj = TilePreprocessor_wrapper(pyrUp_iters)
        return (obj, )


class MeshGraphormerDepthMapPreprocessorProvider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self):
        obj = MeshGraphormerDepthMapPreprocessorProvider_wrapper()
        return (obj, )


NODE_CLASS_MAPPINGS = {
    "OpenPose_Preprocessor_Provider_for_SEGS //Inspire": OpenPose_Preprocessor_Provider_for_SEGS,
    "DWPreprocessor_Provider_for_SEGS //Inspire": DWPreprocessor_Provider_for_SEGS,
    "MiDaS_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": MiDaS_DepthMap_Preprocessor_Provider_for_SEGS,
    "LeRes_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": LeReS_DepthMap_Preprocessor_Provider_for_SEGS,
    # "Zoe_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": Zoe_DepthMap_Preprocessor_Provider_for_SEGS,
    "Canny_Preprocessor_Provider_for_SEGS //Inspire": Canny_Preprocessor_Provider_for_SEGS,
    "MediaPipe_FaceMesh_Preprocessor_Provider_for_SEGS //Inspire": MediaPipe_FaceMesh_Preprocessor_Provider_for_SEGS,
    "HEDPreprocessor_Provider_for_SEGS //Inspire": HEDPreprocessor_Provider_for_SEGS,
    "FakeScribblePreprocessor_Provider_for_SEGS //Inspire": FakeScribblePreprocessor_Provider_for_SEGS,
    "AnimeLineArt_Preprocessor_Provider_for_SEGS //Inspire": AnimeLineArt_Preprocessor_Provider_for_SEGS,
    "Manga2Anime_LineArt_Preprocessor_Provider_for_SEGS //Inspire": Manga2Anime_LineArt_Preprocessor_Provider_for_SEGS,
    "LineArt_Preprocessor_Provider_for_SEGS //Inspire": LineArt_Preprocessor_Provider_for_SEGS,
    "Color_Preprocessor_Provider_for_SEGS //Inspire": Color_Preprocessor_Provider_for_SEGS,
    "InpaintPreprocessor_Provider_for_SEGS //Inspire": InpaintPreprocessor_Provider_for_SEGS,
    "TilePreprocessor_Provider_for_SEGS //Inspire": TilePreprocessor_Provider_for_SEGS,
    "MeshGraphormerDepthMapPreprocessorProvider_for_SEGS //Inspire": MeshGraphormerDepthMapPreprocessorProvider_for_SEGS,
    "MediaPipeFaceMeshDetectorProvider //Inspire": MediaPipeFaceMeshDetectorProvider,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPose_Preprocessor_Provider_for_SEGS //Inspire": "OpenPose Preprocessor Provider (SEGS)",
    "DWPreprocessor_Provider_for_SEGS //Inspire": "DWPreprocessor Provider (SEGS)",
    "MiDaS_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": "MiDaS Depth Map Preprocessor Provider (SEGS)",
    "LeRes_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": "LeReS Depth Map Preprocessor Provider (SEGS)",
    # "Zoe_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": "Zoe Depth Map Preprocessor Provider (SEGS)",
    "Canny_Preprocessor_Provider_for_SEGS //Inspire": "Canny Preprocessor Provider (SEGS)",
    "MediaPipe_FaceMesh_Preprocessor_Provider_for_SEGS //Inspire": "MediaPipe FaceMesh Preprocessor Provider (SEGS)",
    "HEDPreprocessor_Provider_for_SEGS //Inspire": "HED Preprocessor Provider (SEGS)",
    "FakeScribblePreprocessor_Provider_for_SEGS //Inspire": "Fake Scribble Preprocessor Provider (SEGS)",
    "AnimeLineArt_Preprocessor_Provider_for_SEGS //Inspire": "AnimeLineArt Preprocessor Provider (SEGS)",
    "Manga2Anime_LineArt_Preprocessor_Provider_for_SEGS //Inspire": "Manga2Anime LineArt Preprocessor Provider (SEGS)",
    "LineArt_Preprocessor_Provider_for_SEGS //Inspire": "LineArt Preprocessor Provider (SEGS)",
    "Color_Preprocessor_Provider_for_SEGS //Inspire": "Color Preprocessor Provider (SEGS)",
    "InpaintPreprocessor_Provider_for_SEGS //Inspire": "Inpaint Preprocessor Provider (SEGS)",
    "TilePreprocessor_Provider_for_SEGS //Inspire": "Tile Preprocessor Provider (SEGS)",
    "MeshGraphormerDepthMapPreprocessorProvider_for_SEGS //Inspire": "MeshGraphormer Depth Map Preprocessor Provider (SEGS)",
    "MediaPipeFaceMeshDetectorProvider //Inspire": "MediaPipeFaceMesh Detector Provider",
}
