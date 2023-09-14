import { api } from "../../scripts/api.js";

function globalSeedHandler(event) {
	let nodes = app.graph._nodes_by_id;

	for(let i in nodes) {
	    let node = nodes[i];

	    if(node.type == 'GlobalSeed //Inspire') {
	        if(node.widgets) {
			    const w = node.widgets.find((w) => w.name == 'value');
			    w.value = event.detail.value;
	        }
	    }
        else
            if(node.widgets) {
                const w = node.widgets.find((w) => w.name == 'seed' && w.type == 'number');
                if(w) {
                   w.value = event.detail.value;
                }
            }
	}
}

api.addEventListener("inspire-global-seed", globalSeedHandler);


const original_queuePrompt = api.queuePrompt;
async function queuePrompt_with_seed(number, { output, workflow }) {
	workflow.seed_widgets = {};

	for(let i in workflow.nodes) {
	    let node_id = workflow.nodes[i].id;
		let widgets = app.graph._nodes_by_id[node_id].widgets;
		if(widgets) {
		    for(let j in widgets) {
		        if(widgets[j].name == 'seed')
		            workflow.seed_widgets[node_id] = parseInt(j);
		    }
        }
	}

	return await original_queuePrompt.call(api, number, { output, workflow });
}

api.queuePrompt = queuePrompt_with_seed;