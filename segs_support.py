import nodes

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
    "Canny_Preprocessor_Provider_for_SEGS //Inspire": Canny_Preprocessor_Provider_for_SEGS
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPose_Preprocessor_Provider_for_SEGS //Inspire": "OpenPose Preprocessor Provider (SEGS)",
    "Canny_Preprocessor_Provider_for_SEGS //Inspire": "Canny Preprocessor Provider (SEGS)"
}
