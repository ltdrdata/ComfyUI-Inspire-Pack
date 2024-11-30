import comfy
import nodes
from . import utils

SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', "GITS[coeff=1.2]"]


def impact_sampling(*args, **kwargs):
    if 'RegionalSampler' not in nodes.NODE_CLASS_MAPPINGS:
        utils.try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Impact-Pack',
                                      "'Impact Pack' extension is required.")
        raise Exception(f"[ERROR] You need to install 'ComfyUI-Impact-Pack'")

    return nodes.NODE_CLASS_MAPPINGS['RegionalSampler'].separated_sample(*args, **kwargs)


changed_count_cache = {}
changed_cache = {}


def changed_value(uid):
    v = changed_count_cache.get(uid, 0)
    changed_count_cache[uid] = v + 1
    return v + 1


def not_changed_value(uid):
    return changed_count_cache.get(uid, 0)


def is_changed(uid, value):
    if uid not in changed_cache or changed_cache[uid] != value:
        res = changed_value(uid)
    else:
        res = not_changed_value(uid)

    changed_cache[uid] = value

    print(f"keys: {changed_cache.keys()}")

    return res
