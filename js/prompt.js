import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let get_wildcards_list;
let get_wildcard_label;
let is_wildcard_label;
let load_wildcard_status;

try {
	const ImpactPack = await import("../ComfyUI-Impact-Pack/impact-pack.js");
	console.log("[Inspire Pack] Impact Pack module loaded:", ImpactPack);
	get_wildcards_list = ImpactPack.get_wildcards_list;
	get_wildcard_label = ImpactPack.get_wildcard_label;
	is_wildcard_label = ImpactPack.is_wildcard_label;
	load_wildcard_status = ImpactPack.load_wildcard_status;
	console.log("[Inspire Pack] Functions imported:", {
		get_wildcards_list: !!get_wildcards_list,
		get_wildcard_label: !!get_wildcard_label,
		is_wildcard_label: !!is_wildcard_label,
		load_wildcard_status: !!load_wildcard_status
	});
}
catch (error) {
	console.error("[Inspire Pack] Failed to import Impact Pack module:", error);
}

// Fallback for get_wildcards_list
if(!get_wildcards_list) {
	console.warn("[Inspire Pack] get_wildcards_list not available. Using fallback.");
	get_wildcards_list = () => {
		return ["Impact Pack isn't installed or needs browser cache refresh."];
	}
}

// Fallback for on-demand features (backward compatibility with older Impact Pack)
if(!get_wildcard_label) {
	get_wildcard_label = () => { return "Select the Wildcard to add to the text"; };
}
if(!is_wildcard_label) {
	is_wildcard_label = (value) => { return value === "Select the Wildcard to add to the text"; };
}
if(!load_wildcard_status) {
	load_wildcard_status = async () => {}; // No-op for older versions
}

let pb_cache = {};

async function get_prompt_builder_items(category) {
	if(pb_cache[category])
		return pb_cache[category];
	else {
		let res = await api.fetchApi(`/inspire/prompt_builder?category=${category}`);
		let data = await res.json();
		pb_cache[category] = data.presets;
		return data.presets;
	}
}


app.registerExtension({
	name: "Comfy.Inspire.Prompts",

	nodeCreated(node, app) {
		if(node.comfyClass == "WildcardEncode //Inspire") {
			const wildcard_text_widget_index = node.widgets.findIndex((w) => w.name == 'wildcard_text');
			const populated_text_widget_index = node.widgets.findIndex((w) => w.name == 'populated_text');
			const mode_widget_index = node.widgets.findIndex((w) => w.name == 'mode');

			const wildcard_text_widget = node.widgets[wildcard_text_widget_index];
			const populated_text_widget = node.widgets[populated_text_widget_index];

			// lora selector, wildcard selector
			let combo_id = 5;

			// lora
            // The page also calls a callback when it loads a workflow in API format, and on a pick it hands over the label or the picked item depending on the renderer.
            node.widgets[combo_id].callback = () => {
                let lora_name = node._value;
                node._value = undefined;
                if(lora_name === undefined)
                    return;

                if(lora_name.endsWith('.safetensors')) {
                    lora_name = lora_name.slice(0, -12);
                }

                wildcard_text_widget.value += `<lora:${lora_name}>`;
            }

            Object.defineProperty(node.widgets[combo_id], "value", {
                set: (value) => {
                        if (value !== "Select the LoRA to add to the text")
                            node._value = value;
                    },

                get: () => { return "Select the LoRA to add to the text"; }
            });

            // wildcard
            node.widgets[combo_id+1].callback = async () => {
                    const wildcard = node._wildcard_value;
                    node._wildcard_value = undefined;
                    if(wildcard === undefined)
                        return;

                    if(wildcard_text_widget.value != '')
                        wildcard_text_widget.value += ', '

                    wildcard_text_widget.value += wildcard;

                    // Reload wildcard status to update loaded count (Impact Pack staged feature)
                    await load_wildcard_status();
                    app.canvas.setDirty(true);
            }

			Object.defineProperty(node.widgets[combo_id+1], "value", {
				set: (value) => {
                    if (!is_wildcard_label(value))
                        node._wildcard_value = value;
                },
				get: () => { return get_wildcard_label(); }
			});

			Object.defineProperty(node.widgets[combo_id+1].options, "values", {
				set: (x) => {},
				get: () => {
					return get_wildcards_list();
				}
			});

			// Preventing validation errors from occurring in any situation.
			node.widgets[combo_id].serializeValue = () => { return "Select the LoRA to add to the text"; }
			node.widgets[combo_id+1].serializeValue = () => {
				// Always serialize as the default label (not the dynamic on-demand label)
				return "Select the Wildcard to add to the text";
			}

			// wildcard populating
			populated_text_widget.inputEl.disabled = true;
			const mode_widget = node.widgets[mode_widget_index];

			// mode combo
			Object.defineProperty(mode_widget, "value", {
				set: (value) => {
						if(value == true)
							node._mode_value = "populate";
						else if(value == false)
							node._mode_value = "fixed";
						else
							node._mode_value = value; // combo value

						populated_text_widget.inputEl.disabled = node._mode_value == 'populate';
					},
				get: () => {
						if(node._mode_value != undefined)
							return node._mode_value;
						else
							return 'populate';
					 }
			});
		}
		else if(node.comfyClass == "MakeBasicPipe //Inspire") {
			const pos_wildcard_text_widget = node.widgets.find((w) => w.name == 'positive_wildcard_text');
			const pos_populated_text_widget = node.widgets.find((w) => w.name == 'positive_populated_text');
			const neg_wildcard_text_widget = node.widgets.find((w) => w.name == 'negative_wildcard_text');
			const neg_populated_text_widget = node.widgets.find((w) => w.name == 'negative_populated_text');

			const mode_widget = node.widgets.find((w) => w.name == 'wildcard_mode');
			const direction_widget = node.widgets.find((w) => w.name == 'Add selection to');

			// lora selector, wildcard selector
			let combo_id = 5;

            node.widgets[combo_id].callback = () => {
                let lora_name = node._lora_value;
                node._lora_value = undefined;
                if(lora_name === undefined)
                    return;

                if (lora_name.endsWith('.safetensors')) {
                    lora_name = lora_name.slice(0, -12);
                }

                if(direction_widget.value) {
                    pos_wildcard_text_widget.value += `<lora:${lora_name}>`;
                }
                else {
                    neg_wildcard_text_widget.value += `<lora:${lora_name}>`;
                }
            }
			Object.defineProperty(node.widgets[combo_id], "value", {
				set: (value) => {
                        if (value !== "Select the LoRA to add to the text")
                            node._lora_value = value;
					},
				get: () => { return "Select the LoRA to add to the text"; }
			});

            node.widgets[combo_id+1].callback = async () => {
                const wildcard = node._wildcard_value;
                node._wildcard_value = undefined;
                if(wildcard === undefined)
                    return;

                let w = null;
                if(direction_widget.value) {
                    w = pos_wildcard_text_widget;
                }
                else {
                    w = neg_wildcard_text_widget;
                }

                if(w.value != '')
                    w.value += ', '

                w.value += wildcard;

                // Reload wildcard status to update loaded count (Impact Pack staged feature)
                await load_wildcard_status();
                app.canvas.setDirty(true);
            }

			Object.defineProperty(node.widgets[combo_id+1], "value", {
				set: (value) => {
                        if (!is_wildcard_label(value))
                            node._wildcard_value = value;
					},
				get: () => { return get_wildcard_label(); }
			});

			Object.defineProperty(node.widgets[combo_id+1].options, "values", {
				set: (x) => {},
				get: () => {
					return get_wildcards_list();
				}
			});

			// Preventing validation errors from occurring in any situation.
			node.widgets[combo_id].serializeValue = () => { return "Select the LoRA to add to the text"; }
			node.widgets[combo_id+1].serializeValue = () => {
				// Always serialize as the default label (not the dynamic on-demand label)
				return "Select the Wildcard to add to the text";
			}

			// wildcard populating
			pos_populated_text_widget.inputEl.disabled = true;
			neg_populated_text_widget.inputEl.disabled = true;

			// mode combo
			Object.defineProperty(mode_widget, "value", {
				set: (value) => {
						if(value == true)
							node._mode_value = "populate";
						else if(value == false)
							node._mode_value = "fixed";
						else
							node._mode_value = value; // combo value

						pos_populated_text_widget.inputEl.disabled = node._mode_value == 'populate';
						neg_populated_text_widget.inputEl.disabled = node._mode_value == 'populate';
					},
				get: () => {
						if(node._mode_value != undefined)
							return node._mode_value;
						else
							return 'populate';
					 }
			});
		}
		else if(node.comfyClass == "PromptBuilder //Inspire") {
			const preset_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'preset')];
			const category_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'category')];

			Object.defineProperty(preset_widget.options, "values", {
				set: (x) => {},
				get: () => {
					get_prompt_builder_items(category_widget.value);
					if(pb_cache[category_widget.value] == undefined) {
						return ["#PRESET"];
					}
					return pb_cache[category_widget.value];
				}
			});

            preset_widget.callback = () => {
                const preset = node._preset_value;
                node._preset_value = undefined;
                if(preset === undefined)
                    return;

                if(node.widgets[2].value) {
                    node.widgets[2].value += ', ';
                }

                const y = preset.split(':');
                if(y.length == 2)
                    node.widgets[2].value += y[1].trim();
                else
                    node.widgets[2].value += preset.trim();
            }

			Object.defineProperty(preset_widget, "value", {
				set: (value) => {
                    if (value !== "#PRESET")
                        node._preset_value = value;
				},
				get: () => { return '#PRESET'; }
			});

			preset_widget.serializeValue = (workflowNode, widgetIndex) => { return "#PRESET"; };
		}
		else if(node.comfyClass == "SeedExplorer //Inspire"
				|| node.comfyClass == "RegionalSeedExplorerMask //Inspire"
				|| node.comfyClass == "RegionalSeedExplorerColorMask //Inspire") {
			const prompt_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'seed_prompt')];
			const seed_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'additional_seed')];
			const strength_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'additional_strength')];

			let allow_init_seed = node.comfyClass == "SeedExplorer //Inspire";

			node.addWidget("button", "Add to prompt", null, () => {
				if(!prompt_widget.value?.trim() && allow_init_seed) {
					prompt_widget.value = ''+seed_widget.value;
				}
				else {
					if(prompt_widget.value?.trim())
						prompt_widget.value += ', ';

					prompt_widget.value += `${seed_widget.value}:${strength_widget.value.toFixed(2)}`;
					seed_widget.value += 1;
				}
			});
		}
	}
});




const original_queuePrompt = api.queuePrompt;
function collect_widget_idxs(graph, prefix, widget_idx_map) {
	for(const [key, node] of Object.entries(graph._nodes_by_id)) {
		const i = prefix + key;
		let widgets = node.widgets;
		if(widgets) {
			for(let j in widgets) {
				if(['seed', 'noise_seed', 'sampler_name', 'scheduler'].includes(widgets[j].name)
					&& widgets[j].type != 'converted-widget') {
					if(widget_idx_map[i] == undefined) {
						widget_idx_map[i] = {};
					}

					widget_idx_map[i][widgets[j].name] = parseInt(j);
				}
			}
		}

		if(node.subgraph)
			collect_widget_idxs(node.subgraph, i + ':', widget_idx_map);
	}
}

async function queuePrompt_with_widget_idxs(number, { output, workflow }, ...args) {
	workflow.widget_idx_map = {};
	collect_widget_idxs(app.graph, '', workflow.widget_idx_map);

	const seeds_updated = global_seed_updated(output);
	const result = await original_queuePrompt.call(api, number, { output, workflow }, ...args);
	await seeds_updated;
	return result;
}

// The server sends the seeds for the next prompt in an event of its own. That event can come after the answer to the request,
// and the page builds the next prompt of a batch from the widgets as soon as it has that answer.
function global_seed_updated(output) {
	if(!Object.values(output).some((node) => node.class_type == 'GlobalSeed //Inspire'))
		return Promise.resolve();

	return new Promise((resolve) => {
		api.addEventListener("inspire-global-seed", resolve, { once: true });
		setTimeout(resolve, 1000);
	});
}

api.queuePrompt = queuePrompt_with_widget_idxs;