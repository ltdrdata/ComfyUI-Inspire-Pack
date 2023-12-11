import { ComfyApp, app } from "../../scripts/app.js";

export function register_concat_conditionings_with_multiplier_node(nodeType, nodeData, app) {
    if (nodeData.name === 'ConcatConditioningsWithMultiplier //Inspire') {
        var input_name = "conditioning";

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = async function (type, index, connected, link_info) {
            if(!link_info || link_info.type != 'CONDITIONING')
                return;

            let conditionings_count = 0;

            let unconnected = [];
            for(let i in this.inputs) {
            	if(i == index)
            		continue;

                let input = this.inputs[i];
                if(input.name.startsWith('conditioning')) {
                    if(input.link == undefined)
                        unconnected.push(i);

                    conditionings_count += 1;
                }
            }

			let self = this;
            function renames() {
                let con_i = 1;

                let rename_map = {};

                for(let i in self.inputs) {
                    let input = self.inputs[i];
                    if(input.name.startsWith('conditioning')) {
                    	let orig_i = Number(input.name.substring(12));
                    	if(orig_i != con_i) {
                    		rename_map[orig_i] = con_i;
                    		input.name = 'conditioning'+con_i;
                    	}
						con_i++;
                    }
                }

                // update multiplier input
				for(let i in self.inputs) {
                    let input = self.inputs[i];
                    if(input.name.startsWith('multiplier')) {
                    	let orig_i = Number(input.name.substring(10));
                    	if(rename_map[orig_i]) {
                    		input.name = 'multiplier'+rename_map[orig_i];
                    	}
                    }
                }

                // update multiplier widget
				for(let i in self.widgets) {
                    let w = self.widgets[i];
                    if(w.name.startsWith('multiplier')) {
                    	let orig_i = Number(w.name.substring(10));
                    	if(rename_map[orig_i]) {
                    		w.name = 'multiplier'+rename_map[orig_i];
                    	}
                    }
                }

                return con_i;
            }

            if(!connected) {
                // remove 1
                let index = Number(this.inputs[unconnected[0]].name.substring(12));
                this.removeInput(unconnected[0]);

                let target_name = `multiplier${index}`;
                // remove strength from slot
                for(let i in this.inputs) {
                    let input = this.inputs[i];
                    if(input.name.startsWith(target_name)) {
                    	if(input.link) {
                    		let link = app.graph.links[input.link];
                        	const node = app.graph.getNodeById(link.origin_id);
                        	let x = node.outputs[link.origin_slot].links.findIndex((w) => w == input.link);
                        	node.outputs[link.origin_slot].links.splice(x, 1);
                    		await this.disconnectInput(i);
                    		app.graph.links.splice(input.link, 1);
						}
                        await this.removeInput(i);
                        break;
                    }
                }

                const widget_index = this.widgets.findIndex((w) => w.name == target_name);
                this.widgets.splice(widget_index, 1);

                let i = renames();
            }
            else {
				const stackTrace = new Error().stack;

				if(stackTrace.includes('loadGraphData'))
					return;

				if(unconnected.length > 1) {
					// remove garbage slots
					unconnected = unconnected.reverse();
					// TODO: ...
				}

                // add 1
                let i = renames();
                this.addInput(`conditioning${i}`, this.outputs[0].type);
                let config = { min: 0, max: 10, step: 0.1, round: 0.01, precision: 2 };
                let widget = await this.addWidget("number", `multiplier${i}`, 1.0, function (v) {
					if (config.round) {
						this.value = Math.round(v/config.round)*config.round;
					} else {
						this.value = v;
					}
				}, config);
            }

            await this.setSize( this.computeSize() );
        }
    }
}