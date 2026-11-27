import logging

from comfy_execution.graph_utils import GraphBuilder, is_link
from .libs.utils import any_typ
from .libs.common import update_node_status, ListWrapper

class FloatRange:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "start": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, 'step': 0.000000001}),
                        "stop": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, 'step': 0.000000001}),
                        "step": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 100.0, 'step': 0.000000001}),
                        "limit": ("INT", {"default": 100, "min": 2, "max": 4096, "step": 1}),
                        "ensure_end": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     }
                }

    RETURN_TYPES = ("FLOAT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/List"

    def doit(self, start, stop, step, limit, ensure_end):
        if start == stop or step == 0:
            return ([start], )

        reverse = False
        if start > stop:
            reverse = True
            start, stop = stop, start

        res = []
        x = start
        last = x
        while x <= stop and limit > 0:
            res.append(x)
            last = x
            limit -= 1
            x += step

        if ensure_end and last != stop:
            if len(res) >= limit:
                res.pop()

            res.append(stop)

        if reverse:
            res.reverse()

        return (res, )


class WorklistToItemList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "item": (any_typ, ),
                     }
                }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("ITEM_LIST",)
    RETURN_NAMES = ("item_list",)

    FUNCTION = "doit"

    DESCRIPTION = "The list in ComfyUI allows for repeated execution of a sub-workflow.\nThis groups these repetitions (a.k.a. list) into a single ITEM_LIST output.\nITEM_LIST can then be used in ForeachList."

    CATEGORY = "InspirePack/List"

    def doit(self, item):
        return (item, )


# Loop nodes are implemented based on BadCafeCode's reference loop implementation
# https://github.com/BadCafeCode/execution-inversion-demo-comfyui/blob/main/flow_control.py

class ForeachListBegin:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "item_list": ("ITEM_LIST", {"tooltip": "ITEM_LIST containing items to be processed iteratively."}),
                     },
                "optional": {
                        "initial_input": (any_typ, {"tooltip": "If initial_input is omitted, the first item in item_list is used as the initial value, and the processing starts from the second item in item_list."}),
                    }
                }

    RETURN_TYPES = ("FOREACH_LIST_CONTROL", "ITEM_LIST", any_typ, any_typ)
    RETURN_NAMES = ("flow_control", "remained_list", "item", "intermediate_output")
    OUTPUT_TOOLTIPS = (
        "Pass ForeachListEnd as is to indicate the end of the iteration.",
        "Output the ITEM_LIST containing the remaining items during the iteration, passing ForeachListEnd as is to indicate the end of the iteration.",
        "Output the current item during the iteration.",
        "Output the intermediate results during the iteration.")

    FUNCTION = "doit"

    DESCRIPTION = "A starting node for performing iterative tasks by retrieving items one by one from the ITEM_LIST.\nGenerate a new intermediate_output using item and intermediate_output as inputs, then connect it to ForeachListEnd.\nNOTE:If initial_input is omitted, the first item in item_list is used as the initial value, and the processing starts from the second item in item_list."

    CATEGORY = "InspirePack/List"

    def doit(self, item_list, initial_input=None):
        if initial_input is None:
            initial_input = item_list[0]
            item_list = item_list[1:]

        if len(item_list) > 0:
            next_list = ListWrapper(item_list[1:])
            next_item = item_list[0]
        else:
            next_list = ListWrapper([])
            next_item = None

        if next_list.aux is None:
            next_list.aux = len(item_list), None

        return "stub", next_list, next_item, initial_input


class ForeachListEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "flow_control": ("FOREACH_LIST_CONTROL", {"rawLink": True, "tooltip": "Directly connect the output of ForeachListBegin, the starting node of the iteration."}),
                        "remained_list": ("ITEM_LIST", {"tooltip":"Directly connect the output of ForeachListBegin, the starting node of the iteration."}),
                        "intermediate_output": (any_typ, {"tooltip":"Connect the intermediate outputs processed within the iteration here."}),
                     },
                "hidden": {
                    "dynprompt": "DYNPROMPT",
                    "unique_id": "UNIQUE_ID",
                    }
                }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("result",)
    OUTPUT_TOOLTIPS = ("This is the final output value.",)

    FUNCTION = "doit"

    DESCRIPTION = "A end node for performing iterative tasks by retrieving items one by one from the ITEM_LIST.\nNOTE:Directly connect the outputs of ForeachListBegin to 'flow_control' and 'remained_list'."

    CATEGORY = "InspirePack/List"

    def explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def doit(self, flow_control, remained_list, intermediate_output, dynprompt, unique_id):
        if hasattr(remained_list, "aux"):
            if remained_list.aux[1] is None:
                remained_list.aux = (remained_list.aux[0], unique_id)

            update_node_status(remained_list.aux[1], f"{(remained_list.aux[0]-len(remained_list))}/{remained_list.aux[0]} steps", (remained_list.aux[0]-len(remained_list))/remained_list.aux[0])
        else:
            logging.warning("[Inspire Pack] ForeachListEnd: `remained_list` did not come from ForeachList.")

        if len(remained_list) == 0:
            return (intermediate_output,)

        # We want to loop
        upstream = {}

        # Get the list of all nodes between the open and close nodes
        self.explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        open_node = flow_control[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        # We'll use the default prefix, but to avoid having node names grow exponentially in size,
        # we'll use "Recurse" for the name of the recursively-generated copy of this node.
        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)

        new_open.set_input("item_list", remained_list)
        new_open.set_input("initial_input", intermediate_output)

        my_clone = graph.lookup_node("Recurse" )
        result = (my_clone.out(0),)

        return {
            "result": result,
            "expand": graph.finalize(),
        }


class DropItems:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "item_list": ("ITEM_LIST", {"tooltip":"Directly connect the output of ForeachListBegin, the starting node of the iteration."}), },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("ITEM_LIST",)
    OUTPUT_TOOLTIPS = ("This is the final output value.",)

    FUNCTION = "doit"

    DESCRIPTION = ""

    CATEGORY = "InspirePack/List"

    def doit(self, item_list):
        l = ListWrapper([])
        if hasattr(item_list, 'aux'):
            l.aux = item_list.aux
        else:
            logging.warning("[Inspire Pack] DropItems: `item_list` did not come from ForeachList.")

        return (l,)


NODE_CLASS_MAPPINGS = {
    "FloatRange //Inspire": FloatRange,
    "WorklistToItemList //Inspire": WorklistToItemList,
    "ForeachListBegin //Inspire": ForeachListBegin,
    "ForeachListEnd //Inspire": ForeachListEnd,
    "DropItems //Inspire": DropItems,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatRange //Inspire": "Float Range (Inspire)",
    "WorklistToItemList //Inspire": "Worklist To Item List (Inspire)",
    "ForeachListBegin //Inspire": "▶Foreach List (Inspire)",
    "ForeachListEnd //Inspire": "Foreach List◀ (Inspire)",
    "DropItems //Inspire": "Drop Items (Inspire)",
}
