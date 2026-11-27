import { api } from "../../scripts/api.js";

function nodeOfPromptId(id) {
	let graph = app.graph;
	let node;
	for(const part of String(id).split(':')) {
		node = graph?._nodes_by_id[part];
		graph = node?.subgraph;
	}
	return node;
}

function nodeFeedbackHandler(event) {
	let node = nodeOfPromptId(event.detail.node_id);
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


function nodeOutputLabelHandler(event) {
	let node = nodeOfPromptId(event.detail.node_id);
	if(node) {
		node.outputs[event.detail.output_idx].label = event.detail.label;
	}
}

api.addEventListener("inspire-node-output-label", nodeOutputLabelHandler);