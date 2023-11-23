import { ComfyApp, app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Comfy.Inspire.Regional",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === 'ApplyRegionalIPAdapters //Inspire') {
			var input_name = "input";
			var base_slot = 0;

			switch(nodeData.name) {
			case 'ApplyRegionalIPAdapters //Inspire':
				input_name = "regional_ipadapter";
				base_slot = 1;
				break;
			}

			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
				if(!link_info || type == 2)
					return;

				if(this.inputs[0].type == '*'){
					const node = app.graph.getNodeById(link_info.origin_id);
					let origin_type = node.outputs[link_info.origin_slot].type;

					if(origin_type == '*') {
						this.disconnectInput(link_info.target_slot);
						return;
					}

					for(let i in this.inputs) {
						let input_i = this.inputs[i];
						if(input_i.name != 'select' && input_i.name != 'sel_mode')
							input_i.type = origin_type;
					}
				}

				if (!connected && (this.inputs.length > base_slot+1)) {
					const stackTrace = new Error().stack;

					if(
						!stackTrace.includes('LGraphNode.prototype.connect') && // for touch device
						!stackTrace.includes('LGraphNode.connect') && // for mouse device
						!stackTrace.includes('loadGraphData')) {
							this.removeInput(index);
						}
				}

				let slot_i = 1;
				for (let i = base_slot; i < this.inputs.length; i++) {
					let input_i = this.inputs[i];
					input_i.name = `${input_name}${slot_i}`
					slot_i++;
				}

				let last_slot = this.inputs[this.inputs.length - 1];
				if (last_slot.link != undefined) {
					this.addInput(`${input_name}${slot_i}`, this.inputs[base_slot].type);
				}
			}
		}
	}});