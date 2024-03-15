import { api } from "../../scripts/api.js";

async function refresh_data(node) {
	let response = await api.fetchApi('/inspire/cache/list');
	node.widgets[0].value = await response.text();
}

async function remove_key(node, key) {
	await api.fetchApi(`/inspire/cache/remove?key=${key}`);
	node.widgets[1].value = '';
	refresh_data(node);
}

async function clear_data(node) {
	await api.fetchApi('/inspire/cache/clear');
	refresh_data(node);
}

async function set_cache_settings(node) {
	await api.fetchApi('/inspire/cache/settings', {
		method: "POST",
		headers: {"Content-Type": "application/json",},
		body: node.widgets[0].value,
	});
	refresh_data(node);
}

export function register_cache_info(node, app) {
	if(node.comfyClass == "ShowCachedInfo //Inspire") {
		node.addWidget("button", "Remove Key", null, () => { remove_key(node, node.widgets[1].value); });
		node.addWidget("button", "Save Settings", null, () => { set_cache_settings(node); });
		node.addWidget("button", "Refresh", null, () => { refresh_data(node); });
		node.addWidget("button", "Clear", null, () => { clear_data(node); });
		refresh_data(node);
	}
}