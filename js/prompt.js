import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let get_wildcards_list;
try {
	const ImpactPack = await import("../ComfyUI-Impact-Pack/impact-pack.js");
	get_wildcards_list = ImpactPack.get_wildcards_list;
}
catch (error) {}

// fallback
if(!get_wildcards_list) {
	get_wildcards_list = () => { return ["Impact Pack isn't installed or is outdated."]; }
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
            node.widgets[combo_id].callback = (value, canvas, node, pos, e) => {
                let lora_name = node._value;
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
            node.widgets[combo_id+1].callback = (value, canvas, node, pos, e) => {
                    if(wildcard_text_widget.value != '')
                        wildcard_text_widget.value += ', '

                    wildcard_text_widget.value += node._wildcard_value;
            }

			Object.defineProperty(node.widgets[combo_id+1], "value", {
				set: (value) => {
                    if (value !== "Select the Wildcard to add to the text")
                        node._wildcard_value = value;
                },
				get: () => { return "Select the Wildcard to add to the text"; }
			});

			Object.defineProperty(node.widgets[combo_id+1].options, "values", {
				set: (x) => {},
				get: () => {
					return get_wildcards_list();
				}
			});

			// Preventing validation errors from occurring in any situation.
			node.widgets[combo_id].serializeValue = () => { return "Select the LoRA to add to the text"; }
			node.widgets[combo_id+1].serializeValue = () => { return "Select the Wildcard to add to the text"; }

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

            node.widgets[combo_id].callback = (value, canvas, node, pos, e) => {
                let lora_name = node._lora_value;
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

            node.widgets[combo_id+1].callback = (value, canvas, node, pos, e) => {
                let w = null;
                if(direction_widget.value) {
                    w = pos_wildcard_text_widget;
                }
                else {
                    w = neg_wildcard_text_widget;
                }

                if(w.value != '')
                    w.value += ', '

                w.value += node._wildcard_value;
            }

			Object.defineProperty(node.widgets[combo_id+1], "value", {
				set: (value) => {
                        if (value !== "Select the Wildcard to add to the text")
                            node._wildcard_value = value;
					},
				get: () => { return "Select the Wildcard to add to the text"; }
			});

			Object.defineProperty(node.widgets[combo_id+1].options, "values", {
				set: (x) => {},
				get: () => {
					return get_wildcards_list();
				}
			});

			// Preventing validation errors from occurring in any situation.
			node.widgets[combo_id].serializeValue = () => { return "Select the LoRA to add to the text"; }
			node.widgets[combo_id+1].serializeValue = () => { return "Select the Wildcard to add to the text"; }

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

            preset_widget.callback = (value, canvas, node, pos, e) => {
                if(node.widgets[2].value) {
                    node.widgets[2].value += ', ';
                }

                const y = node._preset_value.split(':');
                if(y.length == 2)
                    node.widgets[2].value += y[1].trim();
                else
                    node.widgets[2].value += node._preset_value.trim();
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
async function queuePrompt_with_widget_idxs(number, { output, workflow }, ...args) {
	workflow.widget_idx_map = {};

	for(let i in app.graph._nodes_by_id) {
		let widgets = app.graph._nodes_by_id[i].widgets;
		if(widgets) {
			for(let j in widgets) {
				if(['seed', 'noise_seed', 'sampler_name', 'scheduler'].includes(widgets[j].name)
					&& widgets[j].type != 'converted-widget') {
					if(workflow.widget_idx_map[i] == undefined) {
						workflow.widget_idx_map[i] = {};
					}

					workflow.widget_idx_map[i][widgets[j].name] = parseInt(j);
				}
			}
		}
	}

	return await original_queuePrompt.call(api, number, { output, workflow }, ...args);
}

api.queuePrompt = queuePrompt_with_widget_idxs;