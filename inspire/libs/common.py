import comfy
import nodes
from . import utils

SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD']


def impact_sampling(*args, **kwargs):
    if 'RegionalSampler' not in nodes.NODE_CLASS_MAPPINGS:
        utils.try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Impact-Pack',
                                      "'Impact Pack' extension is required.")
        raise Exception(f"[ERROR] You need to install 'ComfyUI-Impact-Pack'")

    return nodes.NODE_CLASS_MAPPINGS['RegionalSampler'].separated_sample(*args, **kwargs)
