import { ComfyApp, app } from "../../scripts/app.js";

export function register_concat_conditionings_with_multiplier_node(nodeType, nodeData, app) {
	if (nodeData.name === 'ConcatConditioningsWithMultiplier //Inspire') {
		var input_name = "conditioning";

		const onConnectionsChange = nodeType.prototype.onConnectionsChange;
		let this_handler = async function (type, index, connected, link_info) {
			let last_state = this.state_change_handling;
			try {
				this.state_change_handling = true;
				if(!link_info || link_info.type != 'CONDITIONING')
					return;

				let self = this;

				function get_input_count(prefix, linked_only) {
					let cnt = 0;
					for(let i in self.inputs) {
						if(linked_only && !self.inputs[i].link)
							continue;

						if(self.inputs[i].name.startsWith(prefix))
							cnt+=1;
					}

					return cnt;
				}

				function get_widget_count(prefix) {
					let cnt = 0;
					for(let i in self.widgets) {
						if(self.widgets[i].name.startsWith(prefix))
							cnt+=1;
					}

					return cnt;
				}

				function get_unconnected() {
					let unconnected = [];
					for(let i in self.inputs) {
						let input = self.inputs[i];
						if(input.name.startsWith('conditioning')) {
							if(input.link == undefined)
								unconnected.push(i);
						}
					}

					return unconnected;
				}

				let unconnected = get_unconnected();

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

				function remove_multiplier_link(i, link_id) {
						let link = app.graph.links[link_id];
						const node = app.graph.getNodeById(link.origin_id);
						let x = node.outputs[link.origin_slot].links.findIndex((w) => w == link_id);
						node.outputs[link.origin_slot].links.splice(x, 1);
						self.disconnectInput(i);
						app.graph.links.splice(link_id, 1);
				}

				async function remove_target_multiplier(target_name) {
					// remove strength from slot
					for(let i in self.inputs) {
						let input = self.inputs[i];
						if(input.name.startsWith(target_name)) {
							if(input.link) {
								remove_multiplier_link(i, input.link);
							}
							await self.removeInput(i);
							break;
						}
					}

					const widget_index = self.widgets.findIndex((w) => w.name == target_name);
					self.widgets.splice(widget_index, 1);
				}

				async function remove_garbage() {
					let unconnected = get_unconnected();

					// remove unconnected conditionings
					while(unconnected.length > 0) {
						let last_one = unconnected.reverse()[0];
						self.removeInput(last_one);
						unconnected = get_unconnected();
					}

					// remove dangling multipliers
					let conds = new Set();
					let muls = new Set();
					for(let i in self.inputs) {
						let input = self.inputs[i];
						if(input.link && input.name.startsWith('conditioning')) {
							let index = Number(input.name.substring(12));
							conds.add(index);
						}
						else if(input.name.startsWith('multiplier')) {
							let index = Number(input.name.substring(10));
							muls.add(index);
						}
					}
					for(let i in self.widgets) {
						let index = Number(self.widgets[i].name.substring(10));
						muls.add(index);
					}

					let dangling_muls = [...muls].filter(x => !conds.has(x));
					while(dangling_muls.length > 0) {
						let remove_target = dangling_muls.pop();
						let target_name = `multiplier${remove_target}`;
						await remove_target_multiplier(target_name);
					}
				}

				async function ensure_multipliers() {
				    if(self.ensuring_multipliers) {
				        return;
				    }
                    try {
                        self.ensuring_multipliers = true;

                        let ncon = get_input_count('conditioning', true);
                        let nmul = get_input_count('multiplier', false) + get_widget_count('multiplier');

                        if(ncon == 0 && nmul == 0)
                            ncon = 1;

                        for(let i = nmul+1; i<=ncon; i++) {
                            let config = { min: 0, max: 10, step: 0.1, round: 0.01, precision: 2 };

                            // NOTE: addWidget trigger calling ensure_multipliers
                            let widget = await self.addWidget("number", `multiplier${i}`, 1.0, function (v) {
                                if (config.round) {
                                    self.value = Math.round(v/config.round)*config.round;
                                } else {
                                    self.value = v;
                                }
                            }, config);
                        }
					}
					finally{
					    self.ensuring_multipliers = null;
					}
				}

                async function recover_multipliers() {
				    if(self.recover_multipliers) {
				        return;
				    }
                    try {
                        self.recover_multipliers = true;
                        for(let i = 1; i<self.widgets_values.length; i++) {
                            let config = { min: 0, max: 10, step: 0.1, round: 0.01, precision: 2 };

                            // NOTE: addWidget trigger calling recover_multipliers
                            let widget = await self.addWidget("number", `multiplier${i+1}`, 1.0, function (v) {
                                if (config.round) {
                                    self.value = Math.round(v/config.round)*config.round;
                                } else {
                                    self.value = v;
                                }
                            }, config);
                        }
					}
					finally{
					    self.recover_multipliers = null;
					}
				}

				async function ensure_inputs() {
					if(get_unconnected() == 0) {
						let con_i = renames();
						self.addInput(`conditioning${con_i}`, self.outputs[0].type);
					}
				}

                const stackTrace = new Error().stack;
				if(!stackTrace.includes('loadGraphData') && !stackTrace.includes('pasteFromClipboard')) {
					await remove_garbage();
					await ensure_inputs();
				}

                if(!stackTrace.includes('loadGraphData')) {
                    await ensure_multipliers();
                }
                else {
                    await recover_multipliers();
                }

				await this.setSize( this.computeSize() );
			}
			finally {
				this.state_change_handling = last_state;
			}
		}

		nodeType.prototype.onConnectionsChange = this_handler;
	}
}

function ensure_splitter_outputs(node, output_name, value, type) {
	if(node.outputs.length != (value + 1)) {
		while(node.outputs.length != (value + 1)) {
			if(node.outputs.length > value + 1) {
				node.removeOutput(node.outputs.length-1);
			}
			else {
				node.addOutput(`output${node.outputs.length+1}`, type);
			}
		}

		for(let i in node.outputs) {
			let output = node.outputs[i];
			output.name = `${output_name} ${parseInt(i)+1}`;
		}

		if(node.outputs[0].label == type || node.outputs[0].label == 'remained')
			delete node.outputs[0].label;


		let last_output = node.outputs[node.outputs.length-1];
		last_output.name = 'remained';
	}
}

export function register_splitter(node, app) {
	if(node.comfyClass === 'ImageBatchSplitter //Inspire' || node.comfyClass === 'LatentBatchSplitter //Inspire') {
		let split_count = node.widgets[0];

		let output_name = 'output';
		let output_type = "*";

		if(node.comfyClass === 'ImageBatchSplitter //Inspire') {
			output_name = 'image';
			output_type = "IMAGE";
		}
		else if(node.comfyClass === 'LatentBatchSplitter //Inspire') {
			output_name = 'latent';
			output_type = "LATENT";
		}

		ensure_splitter_outputs(node, output_name, split_count.value, output_type);

		Object.defineProperty(split_count, "value", {
			set: async function(value) {
				if(value < 0 || value > 50)
					return;

				ensure_splitter_outputs(node, output_name, value, output_type);
			},
			get: function() {
				return node.outputs.length - 1;
			}
		});
	}
}