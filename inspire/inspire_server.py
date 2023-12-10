import random

import nodes
import server
from enum import Enum
from . import prompt_support
from aiohttp import web


@server.PromptServer.instance.routes.get("/inspire/prompt_builder")
def prompt_builder(request):
    result = {"presets": []}

    if "category" in request.rel_url.query:
        category = request.rel_url.query["category"]
        if category in prompt_support.prompt_builder_preset:
            result['presets'] = prompt_support.prompt_builder_preset[category]

    return web.json_response(result)


class SGmode(Enum):
    FIX = 1
    INCR = 2
    DECR = 3
    RAND = 4


class SeedGenerator:
    def __init__(self, base_value, action):
        self.base_value = base_value

        if action == "fixed" or action == "increment" or action == "decrement" or action == "randomize":
            self.action = SGmode.FIX
        elif action == 'increment for each node':
            self.action = SGmode.INCR
        elif action == 'decrement for each node':
            self.action = SGmode.DECR
        elif action == 'randomize for each node':
            self.action = SGmode.RAND

    def next(self):
        seed = self.base_value

        if self.action == SGmode.INCR:
            self.base_value += 1
            if self.base_value > 1125899906842624:
                self.base_value = 0
        elif self.action == SGmode.DECR:
            self.base_value -= 1
            if self.base_value < 0:
                self.base_value = 1125899906842624
        elif self.action == SGmode.RAND:
            self.base_value = random.randint(0, 1125899906842624)

        return seed


def control_seed(v):
    action = v['inputs']['action']
    value = v['inputs']['value']

    if action == 'increment' or action == 'increment for each node':
        value += 1
        if value > 1125899906842624:
            value = 0
    elif action == 'decrement' or action == 'decrement for each node':
        value -= 1
        if value < 0:
            value = 1125899906842624
    elif action == 'randomize' or action == 'randomize for each node':
        value = random.randint(0, 1125899906842624)

    v['inputs']['value'] = value

    return value


def prompt_seed_update(json_data):
    try:
        seed_widget_map = json_data['extra_data']['extra_pnginfo']['workflow']['seed_widgets']
    except:
        return None

    seed_widget_map = json_data['extra_data']['extra_pnginfo']['workflow']['seed_widgets']
    value = None
    mode = None
    node = None
    action = None

    for k, v in json_data['prompt'].items():
        if 'class_type' not in v:
            continue

        cls = v['class_type']
        if cls == 'GlobalSeed //Inspire':
            mode = v['inputs']['mode']
            action = v['inputs']['action']
            value = v['inputs']['value']
            node = k, v

    # control before generated
    if mode is not None and mode:
        value = control_seed(node[1])

    if value is not None:
        seed_generator = SeedGenerator(value, action)

        for k, v in json_data['prompt'].items():
            for k2, v2 in v['inputs'].items():
                if isinstance(v2, str) and '$GlobalSeed.value$' in v2:
                    v['inputs'][k2] = v2.replace('$GlobalSeed.value$', str(value))

            if k not in seed_widget_map:
                continue

            if 'seed' in v['inputs']:
                if isinstance(v['inputs']['seed'], int):
                    v['inputs']['seed'] = seed_generator.next()

            if 'noise_seed' in v['inputs']:
                if isinstance(v['inputs']['noise_seed'], int):
                    v['inputs']['noise_seed'] = seed_generator.next()

            for k2, v2 in v['inputs'].items():
                if isinstance(v2, str) and '$GlobalSeed.value$' in v2:
                    v['inputs'][k2] = v2.replace('$GlobalSeed.value$', str(value))

    # control after generated
    if mode is not None and not mode:
        control_seed(node[1])

    return value is not None


def workflow_seed_update(json_data):
    nodes = json_data['extra_data']['extra_pnginfo']['workflow']['nodes']
    seed_widget_map = json_data['extra_data']['extra_pnginfo']['workflow']['seed_widgets']
    prompt = json_data['prompt']

    updated_seed_map = {}
    value = None
    for node in nodes:
        node_id = str(node['id'])
        if node_id in prompt:
            if node['type'] == 'GlobalSeed //Inspire':
                value = prompt[node_id]['inputs']['value']
                node['widgets_values'][0] = value
            elif node_id in seed_widget_map:
                widget_idx = seed_widget_map[node_id]

                if 'noise_seed' in prompt[node_id]['inputs']:
                    seed = prompt[node_id]['inputs']['noise_seed']
                else:
                    seed = prompt[node_id]['inputs']['seed']

                node['widgets_values'][widget_idx] = seed
                updated_seed_map[node_id] = seed

    server.PromptServer.instance.send_sync("inspire-global-seed", {"id": node_id, "value": value, "seed_map": updated_seed_map})


def workflow_loadimage_update(json_data):
    prompt = json_data['prompt']

    for v in prompt.values():
        if 'class_type' in v and v['class_type'] == 'LoadImage //Inspire':
            v['inputs']['image'] = "#DATA"


def populate_wildcards(json_data):
    prompt = json_data['prompt']

    if 'ImpactWildcardProcessor' in nodes.NODE_CLASS_MAPPINGS:
        wildcard_process = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardProcessor'].process
        updated_widget_values = {}
        for k, v in prompt.items():
            if 'class_type' in v and v['class_type'] == 'WildcardEncode //Inspire':
                inputs = v['inputs']
                if inputs['mode'] and isinstance(inputs['populated_text'], str):
                    if isinstance(inputs['seed'], list):
                        try:
                            input_node = prompt[inputs['seed'][0]]
                            if input_node['class_type'] == 'ImpactInt':
                                input_seed = int(input_node['inputs']['value'])
                                if not isinstance(input_seed, int):
                                    continue
                            else:
                                print(
                                    f"[Impact Pack] Only ImpactInt and Primitive Node are allowed as the seed for '{v['class_type']}'. It will be ignored. ")
                                continue
                        except:
                            continue
                    else:
                        input_seed = int(inputs['seed'])

                    inputs['populated_text'] = wildcard_process(text=inputs['wildcard_text'], seed=input_seed)
                    inputs['mode'] = False

                    server.PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": k, "widget_name": "populated_text", "type": "text", "data": inputs['populated_text']})
                    updated_widget_values[k] = inputs['populated_text']

        if 'extra_data' in json_data and 'extra_pnginfo' in json_data['extra_data']:
            for node in json_data['extra_data']['extra_pnginfo']['workflow']['nodes']:
                key = str(node['id'])
                if key in updated_widget_values:
                    node['widgets_values'][3] = updated_widget_values[key]
                    node['widgets_values'][4] = False


def onprompt(json_data):
    prompt_support.list_counter_map = {}

    is_changed = prompt_seed_update(json_data)
    if is_changed:
        workflow_seed_update(json_data)

    workflow_loadimage_update(json_data)
    populate_wildcards(json_data)

    return json_data


server.PromptServer.instance.add_on_prompt_handler(onprompt)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
