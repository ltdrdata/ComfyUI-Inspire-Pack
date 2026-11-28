# E2E Test Strategy

E2E tests use the real ComfyUI frontend in Google Chrome. They act through the page as a user does: node search,
mouse, keyboard, the file input and the **Run** button. The supported workflow paths are native JSON loading and
direct UI editing.

The tests that need no browser and no server are described in `tests/README.md`. Run those first.

## Test coverage

| Script | Setup through the UI | Assertions |
| --- | --- | --- |
| `tests/test_e2e_widget_catalog.py` | Load the UI-created catalog and pan to each node at 100% zoom | Every registered node exactly once; live widget names, order, and values match the fixed baseline under positional and named restoration |
| `tests/test_e2e_widget_catalog.py`, the recorder cases | Run the recorder against a page whose frontend code renames, reorders or changes a widget of `Float Range (Inspire)`, and against a session that fails late | The recorder refuses the changed page and leaves both fixtures as they were; it publishes only after a session that ended without a page error |
| `tests/test_e2e_ui.py` | A page with intercepted requests | The driver `tests/e2e_ui.py` keeps the trace and the original failure, and fails a run whose page raised a JavaScript error |

`tests/e2e_ui.py` is the driver. `ComfyUI.add_node` types the node ID without its ending ` //Inspire` into node
search, because search does not find every node of the pack by the whole ID.

## Server setup

Use an isolated server with its own port, user directory, input directory and output directory. The tests create
workflows, change page settings and queue prompts: never point them at a server in use. They do not start or stop
the server.

```bash
python main.py --listen 127.0.0.1 --port 18188 --cpu \
  --base-directory /path/to/an/empty/folder --models-directory /path/to/models \
  --database-url 'sqlite:///:memory:'
```

`<base-directory>/custom_nodes` holds the checkout under test as `comfyui-inspire-pack`, and for the catalog
`comfyui-impact-pack` and `comfyui_ipadapter_plus`. No model is loaded by these tests, so `--cpu` is enough.

## The widget catalog

`tests/workflows/all-inspire-nodes.json` is a native workflow that holds every node of the pack once. It was made in
Chrome by node search and exported with the workflow menu, not written as JSON.
`tests/workflows/all-inspire-nodes.widgets.json` holds the name and value of every widget of every node, in order,
as the page drew them. `AGENTS.md` holds the rules for both files.

The catalog needs this from the server:

- **The Impact Pack is loaded, with its wildcards in Full Cache mode**, which is its default for a small wildcard
  folder. The wildcard dropdown of `Wildcard Encode (Inspire)` and `Make Basic Pipe (Inspire)` shows that mode:
  `Select Wildcard 🟢 Full Cache`. The suite checks this first.
- **IPAdapter Plus is loaded.** The `weight_type` of the four Regional IPAdapter nodes lists the weight types of that
  pack. Without it the only option is `IPAdapterPlus is not installed`.
- **`tests/workflows/all-inspire-nodes.input.png` is the first picture of the input directory**, by name. `Prompt
  Extractor (Inspire)` shows the first picture with a preview, and that preview is a widget of the record. The suite
  checks this too, and stops with setup guidance when another picture comes first.

- **The backend cache of the pack is as a server has it at its start**: no key, and the tag sizes of
  `cache_settings.json`. `Show Cached Info (Inspire)` fills its text box from the server when the node is created or
  loaded. A workflow with a cache node, or one of the shared loaders, leaves a key until the server restarts. The suite
  checks this too.

It does not need the model files whose names it holds. A saved name stays in its widget when the server does not
list it. The names are those under which the files are published:

| Widget | Recorded value |
| --- | --- |
| `ckpt_name` of the shared checkpoint loader and of Make Basic Pipe | `dreamshaper_7.safetensors` |
| `stage_b`, `stage_c` | `stable_cascade_stage_b.safetensors`, `stable_cascade_stage_c.safetensors` |
| `lora_name` | `lcm-lora-sdv1-5.safetensors` |
| `model_name` of the shared diffusion model loader | `lotus-depth-d-v-1-1-fp16.safetensors` |
| `model_name1` of the shared text encoder loader | `clip_l.safetensors` |
| `pulid_file` of PuLID Model Helper | `ip-adapter_pulid_sdxl_fp16.safetensors` |
| `instantid_file`, `control_net_name` of InstantID Model Helper | `ip-adapter.bin`, `diffusion_pytorch_model.safetensors` |

Run it:

```bash
cd tests
INSPIRE_E2E_BASE_URL=http://127.0.0.1:18188 INSPIRE_E2E_ARTIFACTS=/tmp/inspire-catalog-e2e \
  python -m pytest test_e2e_widget_catalog.py -v
```

### Adding a node

Before the change, the catalog suite must pass against the unchanged fixtures. Then add the node to the code and
record it on a server whose model folders hold only files with published names, so that no private file name gets
into the fixture, and whose input directory holds only the catalog picture:

```bash
cd tests
python test_e2e_widget_catalog.py --base-url http://127.0.0.1:18188 --artifacts /tmp/inspire-catalog-recording
```

The recorder first loads the existing catalog in both restoration modes and compares every node with the snapshot.
It then adds each new node by node search, reads its widgets, exports the workflow, loads the export in both modes
and only then replaces the two fixtures. A failure at any step leaves them as they were. Read the diff of both files
before you commit: only the new node may appear. Run the catalog suite once more after the update.

Never regenerate, delete or relax the snapshot to make a failing check pass. A workflow loaded by changed or broken
code is not a new source of truth.

## What the catalog does not cover

The catalog shows that a saved workflow still loads with the same widgets. It does not run a node. What a node
returns is covered by the Python tests of `tests/README.md` for the nodes that need no model, and for every node by
a verification harness kept outside the repository, which runs test workflows in Chrome on an isolated server and
compares each preview with a record.
