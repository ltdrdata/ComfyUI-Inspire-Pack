import { api } from "../../scripts/api.js";

function nodeFeedbackHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.id];
	if(node) {
		if(event.detail.type == "text") {
			const w = node.widgets.find((w) => event.detail.widget_name === w.name);
			if(w) {
				w.value = event.detail.data;
			}
		}
	}
}

api.addEventListener("inspire-node-feedback", nodeFeedbackHandler);