import copy

import pytest

MAX_SEED = 2**32 - 1
GS = "GlobalSeed //Inspire"


def submit(pack, prompt, workflow=None):
    data = {"prompt": copy.deepcopy(prompt)}
    if workflow is not None:
        data["extra_data"] = {"extra_pnginfo": {"workflow": copy.deepcopy(workflow)}}
    return pack.server.PromptServer.trigger_on_prompt(pack.prompt_server, data)


def saved(data, node_id):
    return next(node for node in data["extra_data"]["extra_pnginfo"]["workflow"]["nodes"] if node["id"] == node_id)["widgets_values"]


def events(pack, name):
    return [data for event, data in pack.prompt_server.sent if event == name]


def seeded(value, mode, action, last_seed="7"):
    prompt = {
        "1": {"class_type": GS, "inputs": {"value": value, "mode": mode, "action": action, "last_seed": last_seed}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 11, "sampler_name": "euler", "scheduler": "normal"}},
        "4": {"class_type": "RandomNoise", "inputs": {"noise_seed": 12}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 13, "sampler_name": "euler", "scheduler": "normal"}},
        "6": {"class_type": "KSampler", "inputs": {"seed": ["9", 0], "sampler_name": "euler", "scheduler": "normal"}},
        "9": {"class_type": "PrimitiveInt", "inputs": {"value": 99}},
    }
    workflow = {
        "nodes": [{"id": 1, "type": GS, "widgets_values": [value, mode, action, last_seed]},
                  {"id": 3, "type": "KSampler", "widgets_values": [11, "fixed", 20, 8, "euler", "normal", 1]},
                  {"id": 4, "type": "RandomNoise", "widgets_values": [12, "fixed"]},
                  {"id": 5, "type": "KSampler", "widgets_values": [13, "fixed", 20, 8, "euler", "normal", 1]},
                  {"id": 6, "type": "KSampler", "widgets_values": [20, 8, "euler", "normal", 1]},
                  {"id": 9, "type": "PrimitiveInt", "widgets_values": [99, "fixed"]}],
        "widget_idx_map": {"3": {"seed": 0, "sampler_name": 4, "scheduler": 5}, "4": {"noise_seed": 0},
                           "5": {"seed": 0, "sampler_name": 4, "scheduler": 5}, "6": {"sampler_name": 2, "scheduler": 3}},
    }
    return prompt, workflow


def seeds(data):
    prompt = data["prompt"]
    return [prompt["3"]["inputs"]["seed"], prompt["4"]["inputs"]["noise_seed"], prompt["5"]["inputs"]["seed"]]


@pytest.mark.parametrize("mode,action,used,shown", [
    (True, "fixed", 100, 100),
    (False, "fixed", 100, 100),
    (True, "increment", 101, 101),
    (False, "increment", 100, 101),
    (True, "decrement", 99, 99),
    (False, "decrement", 100, 99),
])
def test_global_seed_gives_every_seed_widget_one_value(pack, mode, action, used, shown):
    prompt, workflow = seeded(100, mode, action)
    data = submit(pack, prompt, workflow)
    assert seeds(data) == [used] * 3
    assert data["prompt"]["6"]["inputs"]["seed"] == ["9", 0]
    assert data["prompt"]["1"]["inputs"]["value"] == shown
    assert events(pack, "inspire-global-seed") == [{"value": shown, "seed_map": {"3": used, "4": used, "5": used}}]
    assert [saved(data, 3)[0], saved(data, 4)[0], saved(data, 5)[0]] == [used] * 3
    assert saved(data, 6) == workflow["nodes"][4]["widgets_values"]


@pytest.mark.parametrize("mode,action,used,shown", [
    (True, "increment for each node", [101, 102, 103], 101),
    (False, "increment for each node", [100, 101, 102], 101),
    (True, "decrement for each node", [99, 98, 97], 99),
    (False, "decrement for each node", [100, 99, 98], 99),
])
def test_global_seed_for_each_node_counts_on_in_the_order_of_the_prompt(pack, mode, action, used, shown):
    prompt, workflow = seeded(100, mode, action)
    data = submit(pack, prompt, workflow)
    assert seeds(data) == used
    assert data["prompt"]["1"]["inputs"]["value"] == shown
    assert events(pack, "inspire-global-seed")[0]["seed_map"] == dict(zip(["3", "4", "5"], used))


@pytest.mark.parametrize("value,action,used", [
    (MAX_SEED, "increment", [0, 0, 0]),
    (0, "decrement", [MAX_SEED] * 3),
    (MAX_SEED - 1, "increment for each node", [MAX_SEED, 0, 1]),
    (0, "decrement for each node", [MAX_SEED, MAX_SEED - 1, MAX_SEED - 2]),
])
def test_global_seed_wraps_around_at_the_ends_of_its_range(pack, value, action, used):
    data = submit(pack, *seeded(value, True, action))
    assert seeds(data) == used


def test_global_seed_randomize_gives_one_new_value_to_all(pack):
    data = submit(pack, *seeded(100, True, "randomize"))
    used = seeds(data)
    assert len(set(used)) == 1 and 0 <= used[0] <= MAX_SEED
    assert data["prompt"]["1"]["inputs"]["value"] == used[0]
    again = submit(pack, *seeded(100, True, "randomize"))
    assert seeds(again) != used


def test_global_seed_randomize_for_each_node_gives_each_its_own_value(pack):
    data = submit(pack, *seeded(100, True, "randomize for each node"))
    used = seeds(data)
    assert len(set(used)) == 3 and all(0 <= seed <= MAX_SEED for seed in used)
    assert used[0] == data["prompt"]["1"]["inputs"]["value"]


@pytest.mark.parametrize("mode", [True, False])
def test_the_workflow_saved_with_the_picture_reproduces_the_seed(pack, mode):
    data = submit(pack, *seeded(100, mode, "increment"))
    if mode:
        assert saved(data, 1) == [101, True, "fixed", 100]
    else:
        assert saved(data, 1) == [100, False, "increment", "7"]


def test_global_seed_value_replaces_its_placeholder_in_text(pack):
    prompt, workflow = seeded(100, True, "increment")
    prompt["7"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "seed_$GlobalSeed.value$_x"}}
    data = submit(pack, prompt, workflow)
    assert data["prompt"]["7"]["inputs"]["filename_prefix"] == "seed_101_x"


def test_seeds_are_left_alone_without_a_global_seed(pack):
    prompt, workflow = seeded(100, True, "increment")
    del prompt["1"]
    data = submit(pack, prompt, workflow)
    assert seeds(data) == [11, 12, 13]
    assert events(pack, "inspire-global-seed") == []


def test_a_prompt_without_the_workflow_of_the_page_is_passed_on_unchanged(pack):
    prompt, _ = seeded(100, True, "increment")
    prompt["8"] = {"class_type": "GlobalSampler //Inspire", "inputs": {"sampler_name": "dpmpp_2m", "scheduler": "karras"}}
    data = submit(pack, prompt)
    assert data["prompt"] == prompt
    assert pack.prompt_server.sent == []


def sampled(sampler="dpmpp_2m", scheduler="karras"):
    prompt = {
        "2": {"class_type": "GlobalSampler //Inspire", "inputs": {"sampler_name": sampler, "scheduler": scheduler}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 1, "sampler_name": "euler", "scheduler": "normal"}},
        "4": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 1, "sampler_name": "euler", "scheduler": ["9", 0]}},
    }
    workflow = {
        "nodes": [{"id": 2, "type": "GlobalSampler //Inspire", "widgets_values": [sampler, scheduler]},
                  {"id": 3, "type": "KSampler", "widgets_values": [1, "fixed", 20, 8, "euler", "normal", 1]},
                  {"id": 4, "type": "KSamplerSelect", "widgets_values": ["euler"]},
                  {"id": 5, "type": "KSampler", "widgets_values": [1, "fixed", 20, 8, "euler", 1]},
                  {"id": 8, "widgets_values": []}],
        "widget_idx_map": {"3": {"seed": 0, "sampler_name": 4, "scheduler": 5}, "4": {"sampler_name": 0},
                           "5": {"seed": 0, "sampler_name": 4}},
    }
    return prompt, workflow


def test_global_sampler_sets_the_nodes_that_have_both_widgets(pack):
    data = submit(pack, *sampled())
    prompt = data["prompt"]
    assert (prompt["3"]["inputs"]["sampler_name"], prompt["3"]["inputs"]["scheduler"]) == ("dpmpp_2m", "karras")
    assert saved(data, 3)[4:6] == ["dpmpp_2m", "karras"]
    assert prompt["4"]["inputs"]["sampler_name"] == "euler"
    feedback = [(data["node_id"], data["widget_name"], data["data"]) for data in events(pack, "inspire-node-feedback")]
    assert ("3", "sampler_name", "dpmpp_2m") in feedback and ("3", "scheduler", "karras") in feedback
    assert not [item for item in feedback if item[0] == "4"]


def test_global_sampler_leaves_a_linked_scheduler_linked(pack):
    data = submit(pack, *sampled())
    assert data["prompt"]["5"]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert data["prompt"]["5"]["inputs"]["scheduler"] == ["9", 0]
    assert saved(data, 5) == [1, "fixed", 20, 8, "dpmpp_2m", 1]


def test_a_muted_global_sampler_changes_nothing(pack):
    prompt, workflow = sampled()
    del prompt["2"]
    data = submit(pack, prompt, workflow)
    assert data["prompt"] == prompt
    assert saved(data, 3) == workflow["nodes"][1]["widgets_values"]


def test_global_seed_and_sampler_together_in_a_workflow_with_a_node_without_type(pack):
    prompt, workflow = seeded(100, True, "fixed")
    prompt["2"] = {"class_type": "GlobalSampler //Inspire", "inputs": {"sampler_name": "dpmpp_2m", "scheduler": "karras"}}
    workflow["nodes"] += [{"id": 2, "type": "GlobalSampler //Inspire", "widgets_values": ["dpmpp_2m", "karras"]},
                          {"id": 30, "widgets_values": []}]
    data = submit(pack, prompt, workflow)
    assert seeds(data) == [100] * 3
    assert data["prompt"]["3"]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert data["prompt"]["6"]["inputs"]["scheduler"] == "karras"


def test_load_image_inspire_is_validated_by_its_data_not_by_the_files_of_the_input_folder(pack, comfyui):
    prompt = {
        "1": {"class_type": "LoadImage //Inspire", "inputs": {"image": "unit-not-in-this-input-folder.png", "image_data": "data:image/png;base64,AA=="}},
        "2": {"class_type": "CacheBackendData //Inspire", "inputs": {"key": "unit", "tag": "image", "data": ["1", 0]}},
    }
    assert not comfyui.validate(prompt)
    data = submit(pack, prompt)
    assert data["prompt"]["1"]["inputs"]["image"] == "#DATA"
    assert comfyui.validate(data["prompt"])


def test_prompt_builder_is_validated_whatever_category_the_workflow_holds(pack, comfyui):
    prompt = {
        "1": {"class_type": "PromptBuilder //Inspire", "inputs": {"category": "unit category of another machine", "preset": "#PRESET", "text": "a fox"}},
        "2": {"class_type": "CacheBackendData //Inspire", "inputs": {"key": "unit", "tag": "text", "data": ["1", 0]}},
    }
    assert not comfyui.validate(prompt)
    data = submit(pack, prompt)
    assert data["prompt"]["1"]["inputs"]["category"] == "#PLACEHOLDER"
    assert comfyui.validate(data["prompt"])
    comfyui.execute(data["prompt"], ["2"])
    assert pack.backend_support.cache["unit"] == ("text", (False, "a fox"))


def test_list_counter_starts_again_with_each_prompt(pack):
    counter = pack.prompt_support.ListCounter()
    assert [counter.doit("x", 10, "4")[0] for _ in range(3)] == [10, 11, 12]
    submit(pack, {"4": {"class_type": "ListCounter //Inspire", "inputs": {"signal": ["1", 0], "base_value": 10}}})
    assert counter.doit("x", 10, "4") == (10,)
