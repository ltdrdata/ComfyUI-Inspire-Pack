import { ComfyApp, app } from "../../scripts/app.js";
import { register_concat_conditionings_with_multiplier_node, register_splitter } from "./inspire-flex.js";
import { register_cache_info } from "./inspire-backend.js";
import { register_loop_node } from "./inspire-loop.js";

app.registerExtension({
	name: "Comfy.Inspire",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		await register_concat_conditionings_with_multiplier_node(nodeType, nodeData, app);
		await register_loop_node(nodeType, nodeData, app);
	},

	nodeCreated(node, app) {
		register_cache_info(node, app);
		register_splitter(node, app);
	}
})