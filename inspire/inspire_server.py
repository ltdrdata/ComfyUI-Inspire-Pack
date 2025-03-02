import random

import nodes
import server
from enum import Enum
from . import prompt_support
from aiohttp import web
from . import backend_support
from .libs import common
import logging


max_seed = 2**32 - 1


@server.PromptServer.instance.routes.get("/inspire/prompt_builder")
def prompt_builder(request):
    result = {"presets": []}

    if "category" in request.rel_url.query:
        category = request.rel_url.query["category"]
        if category in prompt_support.prompt_builder_preset:
            result['presets'] = prompt_support.prompt_builder_preset[category]

    return web.json_response(result)


@server.PromptServer.instance.routes.get("/inspire/cache/remove")
def cache_remove(request):
    if "key" in request.rel_url.query:
        key = request.rel_url.query["key"]
        del backend_support.cache[key]

    return web.Response(status=200)


@server.PromptServer.instance.routes.get("/inspire/cache/clear")
def cache_clear(request):
    backend_support.cache.clear()
    return web.Response(status=200)


@server.PromptServer.instance.routes.get("/inspire/cache/list")
def cache_refresh(request):
    return web.Response(text=backend_support.ShowCachedInfo.get_data(), status=200)


@server.PromptServer.instance.routes.post("/inspire/cache/settings")
async def set_cache_settings(request):
    data = await request.text()
    try:
        backend_support.ShowCachedInfo.set_cache_settings(data)
        return web.Response(text='OK', status=200)
    except Exception as e:
        return web.Response(text=f"{e}", status=500)


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
            if self.base_value > max_seed:
                self.base_value = 0
        elif self.action == SGmode.DECR:
            self.base_value -= 1
            if self.base_value < 0:
                self.base_value = max_seed
        elif self.action == SGmode.RAND:
            self.base_value = random.randint(0, max_seed)

        return seed


def control_seed(v):
    action = v['inputs']['action']
    value = v['inputs']['value']

    if action == 'increment' or action == 'increment for each node':
        value += 1
        if value > max_seed:
            value = 0
    elif action == 'decrement' or action == 'decrement for each node':
        value -= 1
        if value < 0:
            value = max_seed
    elif action == 'randomize' or action == 'randomize for each node':
        value = random.randint(0, max_seed)

    v['inputs']['value'] = value

    return value


def prompt_seed_update(json_data):
    try:
        widget_idx_map = json_data['extra_data']['extra_pnginfo']['workflow']['widget_idx_map']
    except Exception:
        return False, None

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

            if k not in widget_idx_map or ('seed' not in widget_idx_map[k] and 'noise_seed' not in widget_idx_map[k]):
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

    return value is not None, mode


def workflow_seed_update(json_data, mode):
    nodes = json_data['extra_data']['extra_pnginfo']['workflow']['nodes']
    widget_idx_map = json_data['extra_data']['extra_pnginfo']['workflow']['widget_idx_map']
    prompt = json_data['prompt']

    updated_seed_map = {}
    value = None
    for node in nodes:
        node_id = str(node['id'])
        if node_id in prompt:
            if node['type'] == 'GlobalSeed //Inspire':
                if mode is True:
                    node['widgets_values'][3] = node['widgets_values'][0]
                    node['widgets_values'][0] = prompt[node_id]['inputs']['value']
                    node['widgets_values'][2] = 'fixed'

                value = prompt[node_id]['inputs']['value']

            elif node_id in widget_idx_map:
                widget_idx = None
                seed = None
                if 'noise_seed' in prompt[node_id]['inputs']:
                    seed = prompt[node_id]['inputs']['noise_seed']
                    widget_idx = widget_idx_map[node_id].get('noise_seed')
                elif 'seed' in prompt[node_id]['inputs']:
                    seed = prompt[node_id]['inputs']['seed']
                    widget_idx = widget_idx_map[node_id].get('seed')

                if widget_idx is not None:
                    node['widgets_values'][widget_idx] = seed
                    updated_seed_map[node_id] = seed

    server.PromptServer.instance.send_sync("inspire-global-seed", {"value": value, "seed_map": updated_seed_map})


def prompt_sampler_update(json_data):
    try:
        widget_idx_map = json_data['extra_data']['extra_pnginfo']['workflow']['widget_idx_map']
    except Exception:
        return None

    nodes = json_data['extra_data']['extra_pnginfo']['workflow']['nodes']
    prompt = json_data['prompt']

    sampler_name = None
    scheduler = None

    for v in prompt.values():
        cls = v.get('class_type')
        if cls == 'GlobalSampler //Inspire':
            sampler_name = v['inputs']['sampler_name']
            scheduler = v['inputs']['scheduler']

    if sampler_name is None:
        return

    for node in nodes:
        cls = node.get('type')
        if cls == 'GlobalSampler //Inspire' or cls is None:
            continue

        node_id = str(node['id'])

        if node_id in prompt and node_id in widget_idx_map:
            sampler_widget_idx = widget_idx_map[node_id].get('sampler_name')
            scheduler_widget_idx = widget_idx_map[node_id].get('scheduler')

            prompt_inputs = prompt[node_id]['inputs']

            if ('sampler_name' in prompt_inputs and 'scheduler' in prompt_inputs and
                    isinstance(prompt_inputs['sampler_name'], str) and 'scheduler' in prompt_inputs):

                if sampler_widget_idx is not None:
                    prompt_inputs['sampler_name'] = sampler_name
                    node['widgets_values'][sampler_widget_idx] = sampler_name
                    server.PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": node_id, "widget_name": 'sampler_name', "type": "text", "data": sampler_name})

                if scheduler_widget_idx is not None:
                    prompt_inputs['scheduler'] = scheduler
                    node['widgets_values'][scheduler_widget_idx] = scheduler
                    server.PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": node_id, "widget_name": 'scheduler', "type": "text", "data": scheduler})


def workflow_loadimage_update(json_data):
    prompt = json_data['prompt']

    for v in prompt.values():
        if 'class_type' in v and v['class_type'] == 'LoadImage //Inspire':
            v['inputs']['image'] = "#DATA"


def populate_wildcards(json_data):
    prompt = json_data['prompt']

    if 'ImpactWildcardProcessor' in nodes.NODE_CLASS_MAPPINGS:
        if not hasattr(nodes.NODE_CLASS_MAPPINGS['ImpactWildcardProcessor'], 'process'):
            logging.warning("[Inspire Pack] Your Impact Pack is outdated. Please update to the latest version.")
            return

        wildcard_process = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardProcessor'].process
        updated_widget_values = {}
        mbp_updated_widget_values = {}
        for k, v in prompt.items():
            if 'class_type' in v and v['class_type'] == 'WildcardEncode //Inspire':
                inputs = v['inputs']

                # legacy adapter
                if isinstance(inputs['mode'], bool):
                    if inputs['mode']:
                        new_mode = 'populate'
                    else:
                        new_mode = 'fixed'

                    inputs['mode'] = new_mode

                if inputs['mode'] == 'populate' and isinstance(inputs['populated_text'], str):
                    if isinstance(inputs['seed'], list):
                        try:
                            input_node = prompt[inputs['seed'][0]]
                            if input_node['class_type'] == 'ImpactInt':
                                input_seed = int(input_node['inputs']['value'])
                                if not isinstance(input_seed, int):
                                    continue
                            if input_node['class_type'] == 'Seed (rgthree)':
                                input_seed = int(input_node['inputs']['seed'])
                                if not isinstance(input_seed, int):
                                    continue
                            else:
                                logging.warning("[Inspire Pack] Only `ImpactInt`, `Seed (rgthree)` and `Primitive` Node are allowed as the seed for '{v['class_type']}'. It will be ignored. ")
                                continue
                        except:
                            continue
                    else:
                        input_seed = int(inputs['seed'])

                    inputs['populated_text'] = wildcard_process(text=inputs['wildcard_text'], seed=input_seed)
                    inputs['mode'] = 'reproduce'

                    server.PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": k, "widget_name": "populated_text", "type": "text", "data": inputs['populated_text']})
                    updated_widget_values[k] = inputs['populated_text']

                if inputs['mode'] == 'reproduce':
                    server.PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": k, "widget_name": "mode", "type": "text", "value": 'populate'})

            elif 'class_type' in v and v['class_type'] == 'MakeBasicPipe //Inspire':
                inputs = v['inputs']
                if inputs['wildcard_mode'] == 'populate' and (isinstance(inputs['positive_populated_text'], str) or isinstance(inputs['negative_populated_text'], str)):
                    if isinstance(inputs['seed'], list):
                        try:
                            input_node = prompt[inputs['seed'][0]]
                            if input_node['class_type'] == 'ImpactInt':
                                input_seed = int(input_node['inputs']['value'])
                                if not isinstance(input_seed, int):
                                    continue
                            if input_node['class_type'] == 'Seed (rgthree)':
                                input_seed = int(input_node['inputs']['seed'])
                                if not isinstance(input_seed, int):
                                    continue
                            else:
                                logging.warning("[Inspire Pack] Only `ImpactInt`, `Seed (rgthree)` and `Primitive` Node are allowed as the seed for '{v['class_type']}'. It will be ignored. ")
                                continue
                        except:
                            continue
                    else:
                        input_seed = int(inputs['seed'])

                    if isinstance(inputs['positive_populated_text'], str):
                        inputs['positive_populated_text'] = wildcard_process(text=inputs['positive_wildcard_text'], seed=input_seed)
                        server.PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": k, "widget_name": "positive_populated_text", "type": "text", "data": inputs['positive_populated_text']})

                    if isinstance(inputs['negative_populated_text'], str):
                        inputs['negative_populated_text'] = wildcard_process(text=inputs['negative_wildcard_text'], seed=input_seed)
                        server.PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": k, "widget_name": "negative_populated_text", "type": "text", "data": inputs['negative_populated_text']})

                    inputs['wildcard_mode'] = 'reproduce'
                    mbp_updated_widget_values[k] = inputs['positive_populated_text'], inputs['negative_populated_text']

                if inputs['wildcard_mode'] == 'reproduce':
                    server.PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": k, "widget_name": "wildcard_mode", "type": "text", "value": 'populate'})

        if 'extra_data' in json_data and 'extra_pnginfo' in json_data['extra_data']:
            extra_pnginfo = json_data['extra_data']['extra_pnginfo']
            if 'workflow' in extra_pnginfo and extra_pnginfo['workflow'] is not None and 'nodes' in extra_pnginfo['workflow']:
                for node in extra_pnginfo['workflow']['nodes']:
                    key = str(node['id'])
                    if key in updated_widget_values:
                        node['widgets_values'][3] = updated_widget_values[key]
                        node['widgets_values'][4] = 'reproduce'
                    if key in mbp_updated_widget_values:
                        node['widgets_values'][7] = mbp_updated_widget_values[key][0]
                        node['widgets_values'][8] = mbp_updated_widget_values[key][1]
                        node['widgets_values'][5] = 'reproduce'


def force_reset_useless_params(json_data):
    prompt = json_data['prompt']

    for k, v in prompt.items():
        if 'class_type' in v and v['class_type'] == 'PromptBuilder //Inspire':
            v['inputs']['category'] = '#PLACEHOLDER'

    return json_data


def clear_unused_node_changed_cache(json_data):
    prompt = json_data['prompt']

    unused = []
    for x in common.changed_cache.keys():
        if x not in prompt:
            unused.append(x)

    for x in unused:
        del common.changed_cache[x]
        del common.changed_count_cache[x]

    return json_data


def onprompt(json_data):
    prompt_support.list_counter_map = {}

    is_changed, mode = prompt_seed_update(json_data)
    if is_changed:
        workflow_seed_update(json_data, mode)

    prompt_sampler_update(json_data)

    workflow_loadimage_update(json_data)
    populate_wildcards(json_data)

    force_reset_useless_params(json_data)
    clear_unused_node_changed_cache(json_data)

    return json_data


server.PromptServer.instance.add_on_prompt_handler(onprompt)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
