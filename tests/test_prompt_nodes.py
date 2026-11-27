import json

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

FILE = """positive:a fox in the snow,
at dawn

negative:blurry
name:fox
-----
positive:a cat
negative:dark
----
positive:a dog
negative:noisy
name:dog
"""


@pytest.fixture
def prompts(pack, tmp_path, monkeypatch):
    (tmp_path / "unit").mkdir()
    (tmp_path / "unit" / "b.txt").write_text(FILE, encoding="utf-8")
    (tmp_path / "unit" / "a.txt").write_text("positive:a tree\nnegative:fog\n", encoding="utf-8")
    (tmp_path / "unit" / "notes.md").write_text("positive:not a prompt file\nnegative:x\n", encoding="utf-8")
    monkeypatch.setitem(pack.folder_paths.folder_names_and_paths, "inspire_prompts", ([str(tmp_path)], {".txt"}))
    return tmp_path


FROM_FILE = [("a fox in the snow,\nat dawn", "blurry", "fox"), ("a cat", "dark", "unit/b.txt"), ("a dog", "noisy", "dog")]


def test_load_prompts_from_file_reads_each_prompt_with_its_name(pack, prompts):
    node = pack.prompt_support.LoadPromptsFromFile
    assert "unit/b.txt" in node.INPUT_TYPES()["required"]["prompt_file"][0]
    assert node.doit("unit/b.txt", load_cap=0, start_index=0) == (FROM_FILE, 3, 0)


@pytest.mark.parametrize("start_index,load_cap,expected,remaining", [
    (1, 0, FROM_FILE[1:], 0),
    (0, 2, FROM_FILE[:2], 1),
    (-1, 0, FROM_FILE[2:], 0),
    (1, 1, FROM_FILE[1:2], 1),
])
def test_load_prompts_from_file_loads_the_slice_asked_for(pack, prompts, start_index, load_cap, expected, remaining):
    node = pack.prompt_support.LoadPromptsFromFile
    assert node.doit("unit/b.txt", load_cap=load_cap, start_index=start_index) == (expected, 3, remaining)


def test_load_prompts_from_file_takes_text_data_in_place_of_the_file(pack, prompts):
    node = pack.prompt_support.LoadPromptsFromFile
    assert node.doit("unit/b.txt", text_data_opt="positive:given\nnegative:none\n", load_cap=0, start_index=0) == ([("given", "none", "unit/b.txt")], 1, 0)


def test_load_prompts_from_dir_reads_the_text_files_in_name_order(pack, prompts):
    node = pack.prompt_support.LoadPromptsFromDir
    assert "unit" in node.INPUT_TYPES()["required"]["prompt_dir"][0]
    loaded, count, _ = node.doit("unit", load_cap=0, start_index=0)
    a = str(prompts / "unit" / "a.txt")
    b = str(prompts / "unit" / "b.txt")
    assert loaded == [("a tree", "fog", a), FROM_FILE[0], (FROM_FILE[1][0], FROM_FILE[1][1], b), FROM_FILE[2]]
    assert count == 4


@pytest.mark.parametrize("index,expected", [(0, FROM_FILE[0]), (2, FROM_FILE[2]), (9, FROM_FILE[2])])
def test_load_single_prompt_picks_one_by_index_and_the_last_beyond_the_end(pack, prompts, index, expected):
    assert pack.prompt_support.LoadSinglePromptFromFile.doit("unit/b.txt", index) == ([expected],)


def test_unzip_gives_back_what_zip_was_given(pack, run):
    zipped = run(pack.prompt_support.ZipPrompt, positive=["a fox", "a cat"], negative=["blurry", "dark"], name_opt=["fox", "cat"])
    assert run(pack.prompt_support.UnzipPrompt, zipped_prompt=zipped[0]) == [["a fox", "a cat"], ["blurry", "dark"], ["fox", "cat"]]
    assert pack.prompt_support.ZipPrompt().doit("a fox", "blurry") == (("a fox", "blurry", ""),)


def picture(path, prompt):
    info = PngInfo()
    info.add_text("prompt", json.dumps(prompt))
    info.add_text("workflow", json.dumps({"nodes": []}))
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)


def test_prompt_extractor_lists_the_text_inputs_of_the_prompt_in_the_picture(pack, tmp_path, monkeypatch):
    monkeypatch.setattr(pack.folder_paths, "input_directory", str(tmp_path))
    picture(tmp_path / "unit-extract.png", {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fox", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
        "4": {"class_type": "unit node of a pack that is not installed", "inputs": {"text": "x"}},
    })
    pack.prompt_support.PromptExtractor().doit("unit-extract.png", "6.text", "7.text", "", "12")
    assert pack.prompt_server.sent == [("inspire-node-feedback", {
        "node_id": "12", "widget_name": "info", "type": "text",
        "data": "6.text [CLIPTextEncode] ==> a fox\n7.text [CLIPTextEncode] ==> blurry\n"})]


def test_prompt_extractor_says_so_when_the_picture_holds_no_prompt(pack, tmp_path, monkeypatch):
    monkeypatch.setattr(pack.folder_paths, "input_directory", str(tmp_path))
    Image.new("RGB", (8, 8)).save(tmp_path / "unit-plain.png")
    assert pack.prompt_support.PromptExtractor().doit("unit-plain.png", "6.text", "7.text", "", "12") == ("", "")
    assert pack.prompt_server.sent[0][1]["data"] == "There is no prompt information within the image."


@pytest.mark.parametrize("limit,expected", [(3, "9\n8\n7"), (0, "9\n8\n7\n6\n5")])
def test_seed_logger_puts_the_new_seed_on_top(pack, limit, expected):
    assert pack.prompt_support.SeedLogger().doit(9, "8\n7\n6\n5", limit, "3") == {}
    assert pack.prompt_server.sent == [("inspire-node-feedback", {"node_id": "3", "widget_name": "seeds", "type": "text", "data": expected})]
