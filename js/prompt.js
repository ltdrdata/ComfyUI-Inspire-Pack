import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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

			Object.defineProperty(node.widgets[combo_id+1], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(stackTrace.includes('inner_value_change')) {
                            if(value != "Select the Wildcard to add to the text") {
                                if(wildcard_text_widget.value != '')
                                    wildcard_text_widget.value += ', '

	                            wildcard_text_widget.value += value;
                            }
                        }

						node._wvalue = value;
					},
				get: () => {
                        return node._wvalue;
					 }
			});

			Object.defineProperty(node.widgets[combo_id], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(stackTrace.includes('inner_value_change')) {
                            if(value != "Select the LoRA to add to the text") {
	                            let lora_name = value;
	                            if (lora_name.endsWith('.safetensors')) {
	                                lora_name = lora_name.slice(0, -12);
	                            }

	                            wildcard_text_widget.value += `<lora:${lora_name}>`;
	                            if(node.widgets_values) {
	                                node.widgets_values[wildcard_text_widget_index] = wildcard_text_widget.value;
                                }
                            }
                        }

						node._value = value;
					},
				get: () => {
                        return node._value;
					 }
			});

			// Preventing validation errors from occurring in any situation.
			node.widgets[combo_id].serializeValue = () => { return "Select the LoRA to add to the text"; }
			node.widgets[combo_id+1].serializeValue = () => { return "Select the Wildcard to add to the text"; }

			// wildcard populating
			populated_text_widget.inputEl.disabled = true;
            let populate_getter = populated_text_widget.__lookupGetter__('value');
            let populate_setter = populated_text_widget.__lookupSetter__('value');

			const mode_widget = node.widgets[mode_widget_index];
			const seed_widget = node.widgets.find((w) => w.name == 'seed');

			let force_serializeValue = async (n,i) =>
				{
					if(!mode_widget.value) {
						return populated_text_widget.value;
					}
					else {
				        let wildcard_text = await wildcard_text_widget.serializeValue();

						let response = await api.fetchApi(`/impact/wildcards`, {
																method: 'POST',
																headers: { 'Content-Type': 'application/json' },
																body: JSON.stringify({text: wildcard_text, seed: seed_widget.value})
															});

						let populated = await response.json();

						n.widgets_values[mode_widget_index] = false;
						n.widgets_values[populated_text_widget_index] = populated.text;
						populate_setter.call(populated_text_widget, populated.text);

						return populated.text;
					}
				};

			// mode combo
			Object.defineProperty(mode_widget, "value", {
				set: (value) => {
						node._mode_value = value == true || value == "Populate";
						populated_text_widget.inputEl.disabled = value == true || value == "Populate";
					},
				get: () => {
						if(node._mode_value != undefined)
							return node._mode_value;
						else
							return true;
					 }
			});

            // to avoid conflict with presetText.js of pythongosssss
			Object.defineProperty(populated_text_widget, "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(!stackTrace.includes('serializeValue'))
				            populate_setter.call(populated_text_widget, value);
					},
				get: () => {
				        return populate_getter.call(populated_text_widget);
					 }
			});

            wildcard_text_widget.serializeValue = (n,i) => {
                if(node.inputs) {
	                let link_id = node.inputs.find(x => x.name=="wildcard_text")?.link;
	                if(link_id != undefined) {
	                    let link = app.graph.links[link_id];
	                    let input_widget = app.graph._nodes_by_id[link.origin_id].widgets[link.origin_slot];
	                    if(input_widget.type == "customtext") {
	                        return input_widget.value;
	                    }
	                }
	                else {
	                    return wildcard_text_widget.value;
	                }
                }
                else {
                    return wildcard_text_widget.value;
                }
            };

            populated_text_widget.serializeValue = force_serializeValue;
		}
		else if(node.comfyClass == "PromptBuilder //Inspire") {
			const preset_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'preset')];
			var full_preset_widget = preset_widget.options.values;
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

			Object.defineProperty(preset_widget, "value", {
			    set: (x) => {
					const stackTrace = new Error().stack;
					if(stackTrace.includes('inner_value_change')) {
						if(node.widgets[2].value) {
							node.widgets[2].value += ', ';
						}

						const y = x.split(':');
						if(y.length == 2)
							node.widgets[2].value += y[1].trim();
						else
							node.widgets[2].value += x.trim();

						if(node.widgets_values) {
			    			node.widgets_values[2] = node.widgets[2].values;
						}
					};
			    },
			    get: () => { return '#PRESET'; }
			});

			preset_widget.serializeValue = (workflowNode, widgetIndex) => { return "#PRESET"; };
		}
	}
});