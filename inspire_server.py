import random

import server

def control_seed(node_id, v):
    action = v['inputs']['action']
    value = v['inputs']['value']

    if action == 'increment':
        value += 1
    elif action == 'decrement':
        value -= 1
    elif action == 'randomize':
        value = random.randint(0, 1125899906842624)

    v['inputs']['value'] = value
    server.PromptServer.instance.send_sync("inspire-global-seed", {"id": node_id, "value": value})

    return value


def prompt_seed_update(json_data):
    value = None
    mode = None
    node = None

    for k, v in json_data['prompt'].items():
        cls = v['class_type']
        if cls == 'GlobalSeed //Inspire':
            mode = v['inputs']['mode']
            node = k, v

    # control before generated
    if mode is not None and mode:
        value = control_seed(node[0], node[1])

    if value is not None:
        for k, v in json_data['prompt'].items():
            if 'seed' in v['inputs']:
                if isinstance(v['inputs']['seed'], int):
                    v['inputs']['seed'] = value

    # control after generated
    if mode is not None and not mode:
        control_seed(node[0], node[1])

    return value


def workflow_seed_update(json_data, value):
    nodes = json_data['extra_data']['extra_pnginfo']['workflow']['nodes']
    seed_widget_map = json_data['extra_data']['extra_pnginfo']['workflow']['seed_widgets']

    for node in nodes:
        if node['type'] == 'GlobalSeed //Inspire':
            node['widgets_values'][0] = value
        elif str(node['id']) in seed_widget_map:
            node_id = str(node['id'])
            widget_idx = seed_widget_map[node_id]
            node['widgets_values'][widget_idx] = value


def onprompt(json_data):
    value = prompt_seed_update(json_data)
    workflow_seed_update(json_data, value)

    return json_data


server.PromptServer.instance.add_on_prompt_handler(onprompt)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
