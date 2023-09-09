import nodes
import numpy as np

class OpenPose_Preprocessor_wrapper:
    def __init__(self, detect_hand, detect_body, detect_face):
        self.detect_hand = detect_hand
        self.detect_body = detect_body
        self.detect_face = detect_face

    def apply(self, image):
        if 'OpenposePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use OpenPose_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        detect_hand = 'enable' if self.detect_hand else 'disable'
        detect_body = 'enable' if self.detect_body else 'disable'
        detect_face = 'enable' if self.detect_face else 'disable'

        obj = nodes.NODE_CLASS_MAPPINGS['OpenposePreprocessor']()
        return obj.estimate_pose(image, detect_hand, detect_body, detect_face)[0]


class DWPreprocessor_wrapper:
    def __init__(self, detect_hand, detect_body, detect_face):
        self.detect_hand = detect_hand
        self.detect_body = detect_body
        self.detect_face = detect_face

    def apply(self, image):
        if 'DWPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use DWPreprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        detect_hand = 'enable' if self.detect_hand else 'disable'
        detect_body = 'enable' if self.detect_body else 'disable'
        detect_face = 'enable' if self.detect_face else 'disable'

        obj = nodes.NODE_CLASS_MAPPINGS['DWPreprocessor']()
        return obj.estimate_pose(image, detect_hand, detect_body, detect_face)['result'][0]


class LeReS_DepthMap_Preprocessor_wrapper:
    def __init__(self, rm_nearest, rm_background, boost):
        self.rm_nearest = rm_nearest
        self.rm_background = rm_background
        self.boost = boost

    def apply(self, image):
        if 'LeReS-DepthMapPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use LeReS_DepthMap_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        boost = 'enable' if self.boost else 'disable'

        obj = nodes.NODE_CLASS_MAPPINGS['LeReS-DepthMapPreprocessor']()
        return obj.execute(image, self.rm_nearest, self.rm_background, boost=boost)[0]


class MiDaS_DepthMap_Preprocessor_wrapper:
    def __init__(self, a, bg_threshold):
        self.a = a
        self.bg_threshold = bg_threshold

    def apply(self, image):
        if 'MiDaS-DepthMapPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use MiDaS_DepthMap_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['MiDaS-DepthMapPreprocessor']()
        return obj.execute(image, self.a, self.bg_threshold)[0]


class Zoe_DepthMap_Preprocessor_wrapper:
    def apply(self, image):
        if 'Zoe-DepthMapPreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use Zoe_DepthMap_Preprocessor_Provider, you need to install 'ComfyUI's ControlNet Auxiliary Preprocessors.'")

        obj = nodes.NODE_CLASS_MAPPINGS['Zoe-DepthMapPreprocessor']()
        return obj.execute(image)[0]


class Canny_Preprocessor_wrapper:
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def apply(self, image):
        obj = nodes.NODE_CLASS_MAPPINGS['Canny']()
        return obj.detect_edge(image, self.low_threshold, self.high_threshold)[0]


class OpenPose_Preprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detect_hand": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "detect_body": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "detect_face": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, detect_hand, detect_body, detect_face):
        obj = OpenPose_Preprocessor_wrapper(detect_hand, detect_body, detect_face)
        return (obj, )


class DWPreprocessor_Provider_for_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detect_hand": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "detect_body": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "detect_face": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
            }
        }
    RETURN_TYPES = ("SEGS_PREPROCESSOR",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/SEGS/ControlNet"

    def doit(self, detect_hand, detect_body, detect_face):
        obj = DWPreprocessor_wrapper(detect_hand, detect_body, detect_face)
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


NODE_CLASS_MAPPINGS = {
    "OpenPose_Preprocessor_Provider_for_SEGS //Inspire": OpenPose_Preprocessor_Provider_for_SEGS,
    "DWPreprocessor_Provider_for_SEGS //Inspire": DWPreprocessor_Provider_for_SEGS,
    "MiDaS_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": MiDaS_DepthMap_Preprocessor_Provider_for_SEGS,
    "LeRes_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": LeReS_DepthMap_Preprocessor_Provider_for_SEGS,
    # "Zoe_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": Zoe_DepthMap_Preprocessor_Provider_for_SEGS,
    "Canny_Preprocessor_Provider_for_SEGS //Inspire": Canny_Preprocessor_Provider_for_SEGS
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPose_Preprocessor_Provider_for_SEGS //Inspire": "OpenPose Preprocessor Provider (SEGS)",
    "DWPreprocessor_Provider_for_SEGS //Inspire": "DWPreprocessor Provider (SEGS)",
    "MiDaS_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": "MiDaS Depth Map Preprocessor Provider (SEGS)",
    "LeRes_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": "LeReS Depth Map Preprocessor Provider (SEGS)",
    # "Zoe_DepthMap_Preprocessor_Provider_for_SEGS //Inspire": "Zoe Depth Map Preprocessor Provider (SEGS)",
    "Canny_Preprocessor_Provider_for_SEGS //Inspire": "Canny Preprocessor Provider (SEGS)"
}
