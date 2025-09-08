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
function setupInspireNodeFeedback(api) {
	if (!api) {
		console.error("Inspire Pack: Could not find app.api. Node feedback will not work.");
		return;
	}

	api.addEventListener("inspire-node-feedback", ({ detail }) => {
		const node = app.graph.getNodeById(detail.node_id);
		if (!node) return;

		const widget = node.widgets.find(w => w.name === detail.widget_name);
		if (!widget) return;

		const newValue = detail.data !== undefined ? detail.data : detail.value;

		// 1. Update the live value for the UI.
		widget.value = newValue;

		// 2. Update the node's underlying serialization array.
		const widgetIndex = node.widgets.indexOf(widget);
		if (node.widgets_values && widgetIndex > -1) {
			node.widgets_values[widgetIndex] = newValue;
		}

		// 3. Mark the canvas as dirty to ensure saves are correct.
		app.graph.setDirtyCanvas(true, true);

		// Special case: if we updated the mode combo, manually trigger its callback.
		if (widget.name === "wildcard_mode" && widget.callback) {
			widget.callback(newValue);
		}
	});
}

app.registerExtension({
	name: "Comfy.Inspire.Prompts",
	async setup(app) { // The signature only has one argument: 'app'
		// We get the api from the app object and pass it to our helper
		setupInspireNodeFeedback(app.api);
	},
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
		else if (node.comfyClass === "MakeBasicPipe //Inspire") {
			// Find widgets by name for robustness
			const pos_wildcard_text_widget = node.widgets.find((w) => w.name == 'positive_wildcard_text');
			const neg_wildcard_text_widget = node.widgets.find((w) => w.name == 'negative_wildcard_text');
			const pos_populated_text_widget = node.widgets.find((w) => w.name == 'positive_populated_text');
			const neg_populated_text_widget = node.widgets.find((w) => w.name == 'negative_populated_text');
			const mode_widget = node.widgets.find((w) => w.name == 'wildcard_mode');
			const direction_widget = node.widgets.find((w) => w.name == 'Add selection to');
			const lora_widget = node.widgets.find((w) => w.name == 'Select to add LoRA');
			const wildcard_widget = node.widgets.find((w) => w.name == 'Select to add Wildcard');

			// --- Your original LoRA Selector Logic, Preserved ---
			lora_widget.callback = (value, canvas, node, pos, e) => {
				const lora_name = lora_widget._lora_value;
				if (!lora_name) return;
				const final_lora_name = lora_name.endsWith('.safetensors') ? lora_name.slice(0, -12) : lora_name;
				const target_widget = direction_widget.value ? pos_wildcard_text_widget : neg_wildcard_text_widget;

				// Add a comma if needed
				if (target_widget.value.trim() !== '' && !target_widget.value.trim().endsWith(',')) {
					target_widget.value += ', ';
				}
				target_widget.value += `<lora:${final_lora_name}:1.0>`;
				app.graph.setDirtyCanvas(true, true);
			}
			Object.defineProperty(lora_widget, "value", {
				set: (value) => { lora_widget._lora_value = (value !== "Select the LoRA to add to the text") ? value : null; },
				get: () => "Select the LoRA to add to the text"
			});

			wildcard_widget.callback = (value, canvas, node, pos, e) => {
				const wildcard_name = wildcard_widget._wildcard_value;
				if (!wildcard_name) return;
				const target_widget = direction_widget.value ? pos_wildcard_text_widget : neg_wildcard_text_widget;

				// Add a comma if needed
				if (target_widget.value.trim() !== '' && !target_widget.value.trim().endsWith(',')) {
					target_widget.value += ', ';
				}
				// Add the wildcard with standard syntax
				target_widget.value += wildcard_name;
				app.graph.setDirtyCanvas(true, true);
			}
			Object.defineProperty(wildcard_widget, "value", {
				set: (value) => { wildcard_widget._wildcard_value = (value !== "Select the Wildcard to add to the text") ? value : null; },
				get: () => "Select the Wildcard to add to the text"
			});

			Object.defineProperty(wildcard_widget.options, "values", {
				set: (x) => { },
				get: () => {
					// Check if the dependency function exists before calling it
					if (typeof get_wildcards_list === 'function') {
						return ["Select the Wildcard to add to the text"].concat(get_wildcards_list());
					}
					// Fallback if the dependency is not loaded
					return ["Select the Wildcard to add to the text", "---(Wildcard dependency not found)---"];
				}
			});

			// Serialization prevention for combo boxes
			lora_widget.serializeValue = () => "Select the LoRA to add to the text";
			wildcard_widget.serializeValue = () => "Select the Wildcard to add to the text";

			// Mode control logic
			const original_mode_callback = mode_widget.callback;
			mode_widget.callback = function (value) {
				const is_fixed = (value === 'fixed');
				pos_populated_text_widget.inputEl.readOnly = !is_fixed;
				neg_populated_text_widget.inputEl.readOnly = !is_fixed;
				pos_populated_text_widget.inputEl.style.opacity = is_fixed ? 1.0 : 0.6;
				neg_populated_text_widget.inputEl.style.opacity = is_fixed ? 1.0 : 0.6;
				original_mode_callback?.apply(this, arguments);
			}

			// Set initial UI state on load
			setTimeout(() => { if (mode_widget.callback) mode_widget.callback(mode_widget.value); }, 200);
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