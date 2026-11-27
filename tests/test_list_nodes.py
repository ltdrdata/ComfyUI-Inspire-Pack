import pytest
import torch
from PIL import Image

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


@pytest.mark.parametrize("start,stop,step,limit,ensure_end,expected", [
    (0.0, 1.0, 0.25, 100, True, [0.0, 0.25, 0.5, 0.75, 1.0]),
    (0.0, 1.0, 0.3, 100, True, [0.0, 0.3, 0.6, 0.9, 1.0]),
    (0.0, 1.0, 0.3, 100, False, [0.0, 0.3, 0.6, 0.9]),
    (1.0, 0.0, 0.25, 100, True, [1.0, 0.75, 0.5, 0.25, 0.0]),
    (0.0, 1.0, 0.25, 3, True, [0.0, 0.25, 1.0]),
    (0.0, 1.0, 0.25, 3, False, [0.0, 0.25, 0.5]),
    (0.5, 0.5, 0.1, 100, True, [0.5]),
    (0.0, 1.0, 0.0, 100, True, [0.0]),
])
def test_float_range(pack, run, start, stop, step, limit, ensure_end, expected):
    (values,) = run(pack.list_nodes.FloatRange, start=[start], stop=[stop], step=[step], limit=[limit], ensure_end=[ensure_end])
    assert values == pytest.approx(expected)


def pictures(folder):
    for index, color in enumerate(COLORS):
        Image.new("RGB", (8, 8), color).save(folder / f"unit-{index}.png")


def loop(folder, initial=None, drop=False):
    prompt = {
        "1": {"class_type": "LoadImageListFromDir //Inspire", "inputs": {"directory": str(folder), "image_load_cap": 0, "start_index": 0,
                                                                         "load_always": False, "sort_method": "Alphabetical (ASC)"}},
        "2": {"class_type": "WorklistToItemList //Inspire", "inputs": {"item": ["1", 0]}},
        "3": {"class_type": "ForeachListBegin //Inspire", "inputs": {"item_list": ["2", 0]}},
        "4": {"class_type": "ImageBatch", "inputs": {"image1": ["3", 3], "image2": ["3", 2]}},
        "5": {"class_type": "ForeachListEnd //Inspire", "inputs": {"flow_control": ["3", 0], "remained_list": ["3", 1], "intermediate_output": ["4", 0]}},
        "6": {"class_type": "CacheBackendData //Inspire", "inputs": {"key": "unit loop", "tag": "image", "data": ["5", 0]}},
    }
    if initial is not None:
        prompt["7"] = {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": initial}}
        prompt["3"]["inputs"]["initial_input"] = ["7", 0]
    if drop:
        prompt["8"] = {"class_type": "DropItems //Inspire", "inputs": {"item_list": ["3", 1]}}
        prompt["5"]["inputs"]["remained_list"] = ["8", 0]
    return prompt


def colors(images):
    return [tuple(round(float(value) * 255) for value in image[0, 0]) for image in images]


def test_foreach_list_without_initial_input_starts_from_the_first_item(pack, comfyui, tmp_path):
    pictures(tmp_path)
    comfyui.execute(loop(tmp_path), ["6"])
    assert colors(pack.backend_support.cache["unit loop"][1][1]) == COLORS


def test_foreach_list_with_initial_input_runs_once_for_every_item(pack, comfyui, tmp_path):
    pictures(tmp_path)
    comfyui.execute(loop(tmp_path, initial=0xFFFFFF), ["6"])
    assert colors(pack.backend_support.cache["unit loop"][1][1]) == [(255, 255, 255)] + COLORS


def test_drop_items_ends_the_loop_after_the_current_item(pack, comfyui, tmp_path):
    pictures(tmp_path)
    comfyui.execute(loop(tmp_path, drop=True), ["6"])
    assert colors(pack.backend_support.cache["unit loop"][1][1]) == COLORS[:2]


def images(count):
    return [torch.full((1, 4, 4, 3), index / 10) for index in range(count)]


def test_bind_fills_missing_prompts_with_the_default(pack, run):
    prompts = [("a fox", "blurry", "first"), ("a cat", "dark", "second")]
    out = run(pack.prompt_support.BindImageListPromptList, images=images(3), zipped_prompts=prompts,
              default_positive=["a dog"], default_negative=["noisy"])
    assert [image[0, 0, 0, 0].item() for image in out[0]] == pytest.approx([0.0, 0.1, 0.2])
    assert out[1:3] == [["a fox", "a cat", "a dog"], ["blurry", "dark", "noisy"]]


def test_bind_leaves_out_prompts_beyond_the_images(pack, run):
    prompts = [("a fox", "blurry", "first"), ("a cat", "dark", "second"), ("a dog", "noisy", "third")]
    out = run(pack.prompt_support.BindImageListPromptList, images=images(2), zipped_prompts=prompts,
              default_positive=[""], default_negative=[""])
    assert len(out[0]) == 2
    assert out[1:3] == [["a fox", "a cat"], ["blurry", "dark"]]
