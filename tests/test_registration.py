import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parent / "workflows" / "all-inspire-nodes.widgets.json"


def test_the_pack_registers_the_node_types_of_the_widget_catalog(pack):
    catalog = [node["type"] for node in json.loads(CATALOG.read_text())["nodes"]]
    assert sorted(pack.mappings) == sorted(catalog)


def test_every_node_has_a_display_name_and_describes_its_inputs(pack):
    assert set(pack.display_names) == set(pack.mappings)
    for key, node in pack.mappings.items():
        assert isinstance(node.INPUT_TYPES()["required"], dict), key
