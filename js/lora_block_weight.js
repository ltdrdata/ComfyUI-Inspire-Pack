import { ComfyApp, app } from "../../scripts/app.js";

app.registerExtension({
	name: "Comfy.Inspire.LBW",

	nodeCreated(node, app) {
		if(node.comfyClass == "LoraLoaderBlockWeight //Inspire" || node.comfyClass == "MakeLBW //Inspire") {
		    // category filter
			const lora_names_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'lora_name')];
			var full_lora_list = lora_names_widget.options.values;
			const category_filter_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'category_filter')];

			Object.defineProperty(lora_names_widget.options, "values", {
			    set: (x) => {
			        full_lora_list = x;
			    },
			    get: () => {
			        if(category_filter_widget.value == 'All')
			            return full_lora_list;

                    let l = full_lora_list.filter(x => x.startsWith(category_filter_widget.value));
                    return l;
			    }
			});

		    // vector selector
		    let preset_i = 9;
		    let vector_i = 10;

		    if(node.comfyClass == "MakeLBW //Inspire") {
		        preset_i = 7;
		        vector_i = 8;
		    }

			node._value = "Preset";

            node.widgets[preset_i].callback = (v, canvas, node, pos, e) => {
                node.widgets[vector_i].value = node._value.split(':')[1];
                if(node.widgets_values) {
                    node.widgets_values[vector_i] = node.widgets[preset_i].value;
                }
            }

			Object.defineProperty(node.widgets[preset_i], "value", {
				set: (value) => {
				        if(value != "Preset")
						    node._value = value;
					},
				get: () => {
                        return node._value;
					 }
			});
		}

		if(node.comfyClass == "XY Input: Lora Block Weight //Inspire") {
		    // category filter
            const lora_names_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'lora_name')];
			var full_lora_list = lora_names_widget.options.values;
			const category_filter_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'category_filter')];

			Object.defineProperty(lora_names_widget.options, "values", {
			    set: (x) => {
			        full_lora_list = x;
			    },
			    get: () => {
			        if(category_filter_widget.value == 'All')
			            return full_lora_list;

                    let l = full_lora_list.filter(x => x.startsWith(category_filter_widget.value));
                    return l;
			    }
			});

            // vector selector
		    let preset_i = 9;
		    let vector_i = 10;
			node._value = "Preset";

            node.widgets[preset_i].callback = (v, canvas, node, pos, e) => {
                let value = node._value;
                if(!value.startsWith('@') && node.widgets[vector_i].value != "")
                    node.widgets[vector_i].value += "\n";
                if(value.startsWith('@')) {
                    let spec = value.split(':')[1];
                    var n;
                    var sub_n = null;
                    var block = null;

                    if(isNaN(spec)) {
                        let sub_spec = spec.split(',');

                        if(sub_spec.length != 3) {
                            node.widgets_values[vector_i] = '!! SPEC ERROR !!';
                            node._value = '';
                            return;
                        }

                        n = parseInt(sub_spec[0].trim());
                        sub_n = parseInt(sub_spec[1].trim());
                        block = parseInt(sub_spec[2].trim());
                    }
                    else {
                        n = parseInt(spec.trim());
                    }

                    node.widgets[vector_i].value = "";
                    if(sub_n == null) {
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
                        for(let i=1; i<=sub_n; i++) {
                            var temp = "";
                            for(let j=1; j<=n; j++) {
                                if(temp!='')
                                    temp += ',';

                                if(block!=j)
                                    temp += '0';
                                else {
                                    temp += ' ';
                                    for(let k=1; k<=sub_n; k++) {
                                        if(k==i)
                                            temp += 'A ';
                                        else
                                            temp += '0 ';
                                    }
                                }
                            }

                            node.widgets[vector_i].value += `B${block}.SUB${i}:${temp}\n`;
                        }
                    }
                }
                else {
                    node.widgets[vector_i].value += `${value}/${value.split(':')[0]}`;
                }
                if(node.widgets_values) {
                    node.widgets_values[vector_i] = node.widgets[preset_i].value;
                }
            }

			Object.defineProperty(node.widgets[preset_i], "value", {
				set: (value) => {
				        if(value != 'Preset')
						    node._value = value;
					},
				get: () => {
                        return node._value;
					 }
			});
		}
	}
});