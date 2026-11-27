import colorsys


def hex_to_hsv(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    hue = h * 360

    saturation = s
    value = v

    return hue, saturation, value


class RGB_HexToHSV:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "rgb_hex": ("STRING", {"defaultInput": True}),
                    },
                }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("hue", "saturation", "value")
    FUNCTION = "doit"
    CATEGORY = "InspirePack/Util"

    def doit(self, rgb_hex):
        return hex_to_hsv(rgb_hex)


NODE_CLASS_MAPPINGS = {
    "RGB_HexToHSV //Inspire": RGB_HexToHSV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGB_HexToHSV //Inspire": "RGB Hex To HSV (Inspire)",
}
