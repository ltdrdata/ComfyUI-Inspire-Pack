import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function nodeFeedbackHandler(event) {
	let nodes = app.graph._nodes_by_id;
	let node = nodes[event.detail.id];
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

app.registerExtension({
	name: "Comfy.Impact.Exp",

	nodeCreated(node, app) {
		if(node.comfyClass == "LoraLoaderBlockWeight //Inspire") {
		    let preset_i = 8;
		    let vector_i = 9;
			node._value = "Preset";
			Object.defineProperty(node.widgets[preset_i], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(stackTrace.includes('inner_value_change')) {
                            if(value != "Preset") {
                                node.widgets[vector_i].value = value.split(':')[1];
	                            if(node.widgets_values) {
	                                node.widgets_values[vector_i] = node.widgets[preset_i].value;
                                }
                            }
                        }

						node._value = value;
					},
				get: () => {
                        return node._value;
					 }
			});
		}

		if(node.comfyClass == "XY Input: Lora Block Weight //Inspire") {
		    let preset_i = 8;
		    let vector_i = 9;
			node._value = "Preset";
			Object.defineProperty(node.widgets[preset_i], "value", {
				set: (value) => {
				        const stackTrace = new Error().stack;
                        if(stackTrace.includes('inner_value_change')) {
                            if(value != "Preset") {
                                if(!value.startsWith('@') && node.widgets[vector_i].value != "")
                                    node.widgets[vector_i].value += "\n";
                                if(value.startsWith('@')) {
                                    let n = parseInt(value.split(':')[1]);
                                    node.widgets[vector_i].value = "";
                                    for(let i=1; i<=n; i++) {
                                        var temp = "";
                                        for(let j=1; j<=n; j++) {
                                            if(temp!='')
                                                temp += ',';
                                            if(j==i)
                                                temp += 'A';
                                            else
                                                temp += '0';
                                        }

                                        node.widgets[vector_i].value += `B${i}:${temp}\n`;
                                    }
                                }
                                else {
                                    node.widgets[vector_i].value += `${value}/${value.split(':')[0]}`;
                                }
	                            if(node.widgets_values) {
	                                node.widgets_values[vector_i] = node.widgets[preset_i].value;
                                }
                            }
                        }

						node._value = value;
					},
				get: () => {
                        return node._value;
					 }
			});
		}
	}
});