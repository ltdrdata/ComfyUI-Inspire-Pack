# Tests of the Inspire Pack

Run both groups below before a pull request. They need no browser, no running server and no model.
The Chrome suites at the end need an isolated server and run only when named.

## Python tests

The tests import the code of the pack into a real ComfyUI and run it on the CPU. Use the Python environment of that
ComfyUI, with the requirements of the pack and pytest (`python -m pip install -r tests/requirements.txt`).
Run them from the `tests/` folder:

```bash
cd tests
INSPIRE_COMFY_ROOT=/path/to/ComfyUI python -m pytest -q
```

`INSPIRE_COMFY_ROOT` is the ComfyUI folder that holds `nodes.py` and `comfy/`. Without it the run stops with that message.
Set `INSPIRE_IMPACT_ROOT` to an Impact Pack checkout to also run `test_populate.py`, which needs the wildcard code of the
Impact Pack; without it those cases are skipped and say why.

Only `server.PromptServer.instance` is replaced, by a stand-in in `conftest.py` that keeps the routes, the prompt
handler and the messages the pack sends. Each test gets a fresh stand-in and a fresh backend cache. The `comfyui`
fixture registers the nodes of the pack as ComfyUI does at its start, for ComfyUI's validation and executor.
Importing the pack creates `resources/prompt-builder.yaml` from its example, as the first start of ComfyUI does.

| File | What it checks |
| --- | --- |
| `test_prompt_handlers.py` | The prompt handler, run as the server runs it: Global Seed in each mode and action, the wrap at the ends of the seed range, the workflow saved with the picture, Global Sampler, Load Image (Inspire) and Prompt Builder passing validation, List Counter starting again |
| `test_populate.py` | Wildcard Encode and Make Basic Pipe in populate, fixed and reproduce mode, with the wildcard code of the Impact Pack |
| `test_server_routes.py` | The HTTP routes of `inspire_server.py` through aiohttp's test client: Prompt Builder presets, the cache list, remove, clear and settings |
| `test_noise.py` | RandomNoise (inspire) on the CPU against ComfyUI's RandomNoise, the batch seed modes, the variation, the internal seed, Seed Explorer, Composite Noise |
| `test_lora_block_weight.py` | LoRA Loader (Block Weight), Make and Apply LBW and LoRA Block Info on a tiny made-up model, against ComfyUI's LoRA loader; vectors, A and B, R, inverse, presets, block specs |
| `test_backend_cache.py` | TaggedCache sizes, the cache nodes through ComfyUI's list handling, Is Cached, the shared checkpoint loader reading the cache |
| `test_list_nodes.py` | Float Range, the Foreach List loop and Drop Items run by ComfyUI's executor, Bind [ImageList, PromptList] |
| `test_prompt_nodes.py` | Prompt files from a folder of the test's own, Zip and Unzip Prompt, Prompt Extractor, Seed Logger |
| `test_image_nodes.py` | Load Image (Inspire) from its data, the directory loaders, batch sizes, the splitters, Color Map To Masks, RGB Hex To HSV |
| `test_regional_and_conditioning.py` | Color masks, Color Mask To Depth Mask, the Regional Seed Explorers, Regional CFG, Remove ControlNet, Conditioning Upscale and Stretch |
| `test_segs_preprocessors.py` | NormalMap and DepthAnything Preprocessor Provider (SEGS) with stand-ins for the nodes of ControlNet Auxiliary Preprocessors: the node each choice runs, the widget values and resolution it gets, the request to install the pack when it is missing |
| `test_model_nodes.py` | IPAdapter Model Helper with a preset and the SDXL check, IPAdapter Model Helper [Manual] with the picked files and the folders of `extra_model_paths.yaml`, with stand-ins for the loaders of IPAdapter Plus |
| `test_registration.py` | The pack registers the node types of the widget catalog, each with a display name |

## JavaScript tests

The tests load the shipped files of `js/` as ES modules, with stand-ins for `app`, `api` and the module of the
Impact Pack (`page.mjs`, `page_hooks.mjs`). They need Node 20.6 or newer and nothing else. Run from the repository root:

```bash
node --test tests/*.test.mjs
```

| File | What it checks |
| --- | --- |
| `seed.test.mjs` | `seed.js` writes the seeds the server sent; `prompt.js` sends the widget indices of seeds, samplers and schedulers and holds the next prompt until the seeds have come |
| `prompt.test.mjs` | The LoRA, wildcard and preset combos that add a pick to a text, in both renderers; the mode widget; the Seed Explorer button |
| `lora_block_weight.test.mjs` | The preset combo fills `block_vector`, the category filter of the LoRA list, the test presets of the XY input |
| `inspire_flex.test.mjs` | The outputs the batch splitters grow and the inputs and multipliers of Concat Conditionings with Multiplier |
| `common.test.mjs` | The feedback of the server reaches the widget and the output it names |

## Chrome suites

These drive the page of a running, isolated test server through Chrome with Playwright. Never point them at a server
in use. `python -m pytest` leaves them out unless they are named:

```bash
cd tests
INSPIRE_E2E_BASE_URL=http://127.0.0.1:18188 INSPIRE_E2E_ARTIFACTS=/tmp/inspire-e2e python -m pytest test_e2e_widget_catalog.py test_e2e_ui.py
```

`test_e2e_widget_catalog.py` loads `workflows/all-inspire-nodes.json` and compares the widget names, order and values of
every node with `workflows/all-inspire-nodes.widgets.json`, in positional and in named restoration. Both files are a
fixed reference: do not regenerate them from a running server. The server needs the Impact Pack, IPAdapter Plus and
`workflows/all-inspire-nodes.input.png` as the first picture of its input directory:
[`docs/E2E_TEST_STRATEGY.md`](../docs/E2E_TEST_STRATEGY.md) has the setup and the way to add a node. `test_e2e_ui.py` checks the driver in `e2e_ui.py`
against intercepted requests. Screenshots and traces go to `INSPIRE_E2E_ARTIFACTS`.
