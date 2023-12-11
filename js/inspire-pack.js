import { ComfyApp, app } from "../../scripts/app.js";
import { register_concat_conditionings_with_multiplier_node } from "./inspire-flex.js";

app.registerExtension({
	name: "Comfy.Inspire",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		await register_concat_conditionings_with_multiplier_node(nodeType, nodeData, app);
	}
})