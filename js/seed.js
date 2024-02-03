import { api } from "../../scripts/api.js";

function globalSeedHandler(event) {
	let nodes = app.graph._nodes_by_id;

	for(let i in nodes) {
		let node = nodes[i];

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
				if(w && event.detail.seed_map[node.id] != undefined) {
					w.value = event.detail.seed_map[node.id];
				}
			}
	}
}

api.addEventListener("inspire-global-seed", globalSeedHandler);