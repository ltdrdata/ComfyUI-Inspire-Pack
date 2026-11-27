import comfy
import nodes
from . import utils
import logging
from server import PromptServer


ADDITIONAL_SCHEDULERS = ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan', 'OSS Chroma']


def get_schedulers():
    return list(comfy.samplers.SCHEDULER_HANDLERS) + ADDITIONAL_SCHEDULERS


def impact_sampling(*args, **kwargs):
    if 'RegionalSampler' not in nodes.NODE_CLASS_MAPPINGS:
        utils.try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Impact-Pack',
                                      "'Impact Pack' extension is required.")
        raise Exception("[ERROR] You need to install 'ComfyUI-Impact-Pack'")

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

    logging.info(f"keys: {changed_cache.keys()}")

    return res


def update_node_status(node, text, progress=None):
    if PromptServer.instance.client_id is None:
        return

    PromptServer.instance.send_sync("inspire/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, PromptServer.instance.client_id)


class ListWrapper:
    def __init__(self, data, aux=None):
        if isinstance(data, ListWrapper):
            self._data = data
            if aux is None:
                self.aux = data.aux
            else:
                self.aux = aux
        else:
            self._data = list(data)
            self.aux = aux

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ListWrapper(self._data[index], self.aux)
        else:
            return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"ListWrapper({self._data}, aux={self.aux})"
