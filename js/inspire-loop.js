import { inspireProgressBadge } from "./progress-badge.js"

export function register_loop_node(nodeType, nodeData, app) {
	if(nodeData.name == 'ForeachListEnd //Inspire') {
		inspireProgressBadge.addStatusHandler(nodeType);
	}
}