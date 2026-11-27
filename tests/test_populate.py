from test_prompt_handlers import events, saved, submit

TEXT = "a {red|blue|green|white} fox"
LORA = "Select the LoRA to add to the text"
WILDCARD = "Select the Wildcard to add to the text"


def encode(mode, seed=3, populated="kept {x|y}", text=TEXT):
    return {"class_type": "WildcardEncode //Inspire", "inputs": {
        "model": ["1", 0], "clip": ["1", 1], "token_normalization": "none", "weight_interpretation": "comfy",
        "wildcard_text": text, "populated_text": populated, "mode": mode,
        "Select to add LoRA": LORA, "Select to add Wildcard": WILDCARD, "seed": seed}}


def pipe(mode, seed=3, negative="blurry {a|b|c|d}"):
    return {"class_type": "MakeBasicPipe //Inspire", "inputs": {
        "ckpt_name": "unit.safetensors", "ckpt_key_opt": "", "positive_wildcard_text": TEXT, "negative_wildcard_text": negative,
        "Add selection to": True, "Select to add LoRA": LORA, "Select to add Wildcard": WILDCARD, "wildcard_mode": mode,
        "positive_populated_text": "kept {x|y}", "negative_populated_text": "kept", "token_normalization": "none",
        "weight_interpretation": "comfy", "stop_at_clip_layer": -2, "seed": seed}}


def page_workflow():
    return {"nodes": [{"id": 9, "type": "WildcardEncode //Inspire",
                       "widgets_values": ["none", "comfy", TEXT, "kept {x|y}", "populate", LORA, WILDCARD, 3, "fixed"]}]}


def feedback(pack):
    return [(data["node_id"], data["widget_name"], data.get("data", data.get("value"))) for data in events(pack, "inspire-node-feedback")]


def test_the_seed_decides_the_wildcard_options_that_are_picked(wildcards):
    assert wildcards.process(text=TEXT, seed=3) != wildcards.process(text=TEXT, seed=5)


def test_wildcard_encode_populates_its_text_with_its_seed(pack, wildcards):
    for seed in (3, 5):
        pack.prompt_server.sent.clear()
        data = submit(pack, {"9": encode("populate", seed)}, page_workflow())
        populated = wildcards.process(text=TEXT, seed=seed)
        assert data["prompt"]["9"]["inputs"]["populated_text"] == populated
        assert data["prompt"]["9"]["inputs"]["mode"] == "reproduce"
        assert feedback(pack) == [("9", "populated_text", populated), ("9", "mode", "populate")]
        assert saved(data, 9)[3:5] == [populated, "reproduce"]


def test_wildcard_encode_saved_with_the_old_boolean_mode_populates(pack, wildcards):
    data = submit(pack, {"9": encode(True)})
    assert data["prompt"]["9"]["inputs"]["populated_text"] == wildcards.process(text=TEXT, seed=3)


def test_wildcard_encode_takes_the_seed_of_a_linked_rgthree_seed_node(pack, wildcards):
    prompt = {"4": {"class_type": "Seed (rgthree)", "inputs": {"seed": 5}}, "9": encode("populate", ["4", 0])}
    data = submit(pack, prompt)
    assert data["prompt"]["9"]["inputs"]["populated_text"] == wildcards.process(text=TEXT, seed=5)


def test_wildcard_encode_in_fixed_mode_keeps_its_populated_text(pack, wildcards):
    data = submit(pack, {"9": encode("fixed")}, page_workflow())
    assert data["prompt"]["9"]["inputs"] == encode("fixed")["inputs"]
    assert feedback(pack) == []
    assert saved(data, 9) == page_workflow()["nodes"][0]["widgets_values"]


def test_wildcard_encode_in_reproduce_mode_runs_once_with_its_text_and_turns_the_page_back_to_populate(pack, wildcards):
    data = submit(pack, {"9": encode("reproduce")})
    assert data["prompt"]["9"]["inputs"]["populated_text"] == "kept {x|y}"
    assert feedback(pack) == [("9", "mode", "populate")]


def test_make_basic_pipe_populates_both_texts_with_its_seed(pack, wildcards):
    data = submit(pack, {"9": pipe("populate", 5)})
    inputs = data["prompt"]["9"]["inputs"]
    positive, negative = wildcards.process(text=TEXT, seed=5), wildcards.process(text="blurry {a|b|c|d}", seed=5)
    assert (inputs["positive_populated_text"], inputs["negative_populated_text"]) == (positive, negative)
    assert inputs["wildcard_mode"] == "reproduce"
    assert feedback(pack) == [("9", "positive_populated_text", positive), ("9", "negative_populated_text", negative),
                              ("9", "wildcard_mode", "populate")]


def test_make_basic_pipe_in_fixed_mode_keeps_its_populated_texts(pack, wildcards):
    data = submit(pack, {"9": pipe("fixed")})
    assert data["prompt"]["9"]["inputs"] == pipe("fixed")["inputs"]
    assert feedback(pack) == []
