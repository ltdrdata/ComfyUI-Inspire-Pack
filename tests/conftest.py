import ast
import asyncio
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

collect_ignore = ["test_e2e_ui.py", "test_e2e_widget_catalog.py"]


class PromptServerStandIn:
    def __init__(self, routes, on_prompt_handlers):
        self.routes = routes
        self.on_prompt_handlers = on_prompt_handlers
        self.client_id = None
        self.last_node_id = None
        self.sent = []

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def send_sync(self, event, data, sid=None):
        self.sent.append((event, data))


@pytest.fixture(scope="session")
def imported():
    comfy_root = os.environ.get("INSPIRE_COMFY_ROOT")
    if not comfy_root:
        pytest.fail("Set INSPIRE_COMFY_ROOT to the ComfyUI folder that holds nodes.py and comfy/, "
                    "and run with the Python environment of that ComfyUI.", pytrace=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(comfy_root)
        from comfy.cli_args import args
        patch.setattr(args, "cpu", True)
        from aiohttp import web
        import folder_paths
        import server
        standin = PromptServerStandIn(web.RouteTableDef(), [])
        patch.setattr(server.PromptServer, "instance", standin, raising=False)
        init = ast.parse((ROOT / "__init__.py").read_text())
        node_list = next(ast.literal_eval(node.value) for node in init.body
                         if isinstance(node, ast.Assign) and node.targets[0].id == "node_list")
        modules = {name: importlib.import_module("inspire." + name) for name in node_list}
        assert Path(modules["inspire_server"].__file__).resolve().parents[1] == ROOT
        mappings = {key: node for module in modules.values() for key, node in module.NODE_CLASS_MAPPINGS.items()}
        display_names = {key: name for module in modules.values() for key, name in module.NODE_DISPLAY_NAME_MAPPINGS.items()}
        yield SimpleNamespace(server=server, folder_paths=folder_paths, routes=standin.routes, on_prompt_handlers=standin.on_prompt_handlers,
                              mappings=mappings, display_names=display_names, **modules)


@pytest.fixture
def pack(imported, monkeypatch):
    from inspire.libs import common, utils
    standin = PromptServerStandIn(imported.routes, imported.on_prompt_handlers)
    monkeypatch.setattr(imported.server.PromptServer, "instance", standin)
    backend = imported.backend_support
    monkeypatch.setattr(backend, "cache", utils.TaggedCache(backend.cache_settings))
    monkeypatch.setattr(backend, "cache_count", {})
    monkeypatch.setattr(common, "changed_cache", {})
    monkeypatch.setattr(common, "changed_count_cache", {})
    monkeypatch.setattr(imported.prompt_support, "list_counter_map", {})
    return SimpleNamespace(**vars(imported), prompt_server=standin, utils=utils)


@pytest.fixture
def run(pack):
    import execution

    def run(node, **inputs):
        outputs, _, _, _ = asyncio.run(execution.get_output_data("unit", "1", node(), inputs))
        return outputs
    return run


@pytest.fixture
def comfyui(pack, monkeypatch):
    import execution
    import nodes
    for key, node in pack.mappings.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, key, node)

    def validate(prompt):
        return asyncio.run(execution.validate_prompt("unit", prompt, None))[0]

    def execute(prompt, outputs):
        executor = execution.PromptExecutor(pack.prompt_server, cache_type=execution.CacheType.CLASSIC,
                                            cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
        executor.execute(prompt, "unit", {}, outputs)
        assert executor.success, executor.status_messages[-1]
    return SimpleNamespace(validate=validate, execute=execute)


@pytest.fixture
def wildcards(pack, monkeypatch):
    impact_root = os.environ.get("INSPIRE_IMPACT_ROOT")
    if not impact_root:
        pytest.skip("Set INSPIRE_IMPACT_ROOT to an Impact Pack checkout to run the populate cases.")
    monkeypatch.syspath_prepend(os.path.join(impact_root, "modules"))
    try:
        from impact import wildcards
    except ImportError as error:
        pytest.skip(f"The wildcard code of the Impact Pack at INSPIRE_IMPACT_ROOT cannot be imported: {error}")
    import nodes
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "ImpactWildcardProcessor", SimpleNamespace(process=wildcards.process))
    return wildcards
