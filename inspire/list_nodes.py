class FloatRange:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "start": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, 'step': 0.000000001}),
                        "stop": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, 'step': 0.000000001}),
                        "step": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 100.0, 'step': 0.000000001}),
                        "limit": ("INT", {"default": 100, "min": 2, "max": 4096, "step": 1}),
                        "ensure_end": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     }
                }

    RETURN_TYPES = ("FLOAT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, start, stop, step, limit, ensure_end):
        if start >= stop or step == 0:
            return ([start], )

        res = []
        x = start
        last = x
        while x <= stop and limit > 0:
            res.append(x)
            last = x
            limit -= 1
            x += step

        if ensure_end and last != stop:
            if len(res) >= limit:
                res.pop()

            res.append(stop)

        return (res, )


NODE_CLASS_MAPPINGS = {
    "FloatRange //Inspire": FloatRange,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatRange //Inspire": "Float Range (Inspire)"
}
