import { api } from "../../scripts/api.js";

function updateSeeds(graph, prefix, event) {
	for(const [key, node] of Object.entries(graph._nodes_by_id)) {
		const id = prefix + key;

		if(node.type == 'GlobalSeed //Inspire') {
			if(node.widgets) {
				const w = node.widgets.find((w) => w.name == 'value');
				const last_w = node.widgets.find((w) => w.name == 'last_seed');
				last_w.value = w.value;
				if(event.detail.value != null)
					w.value = event.detail.value;
			}
		}
		else
			if(node.widgets) {
				const w = node.widgets.find((w) => (w.name == 'seed' || w.name == 'noise_seed') && w.type == 'number');
				if(w && event.detail.seed_map[id] != undefined) {
					w.value = event.detail.seed_map[id];
				}
			}

		if(node.subgraph)
			updateSeeds(node.subgraph, id + ':', event);
	}
}

function globalSeedHandler(event) {
	updateSeeds(app.graph, '', event);
}

api.addEventListener("inspire-global-seed", globalSeedHandler);