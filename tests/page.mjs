import { readFileSync } from 'node:fs';
import { register } from 'node:module';

register('./page_hooks.mjs', import.meta.url);

globalThis.inspireTestPages = {};
globalThis.window = globalThis;
let pages = 0;

export async function openPage(frontend, ...files) {
    const page = ++pages;
    globalThis.inspireTestPages[page] = frontend;
    globalThis.app = frontend['/scripts/app.js']?.app;
    for (const file of files)
        await import(new URL(`../js/${file}?page=${page}`, import.meta.url));
}

export function makeApi({ fetchApi = async () => ({ json: async () => ({}) }), queuePrompt = async () => ({}) } = {}) {
    const listeners = {};
    return {
        fetchApi,
        queuePrompt,
        addEventListener(name, listener, options) {
            (listeners[name] ??= []).push({ listener, once: options?.once });
        },
        dispatch(name, detail) {
            const current = listeners[name] ?? [];
            listeners[name] = current.filter(entry => !entry.once);
            for (const { listener } of current)
                listener({ detail });
        },
    };
}

export function makeApp(graph = { _nodes_by_id: {} }) {
    const app = {
        graph,
        extensions: [],
        canvas: { dirty: 0, setDirty() { this.dirty += 1; } },
        registerExtension(extension) { app.extensions.push(extension); },
    };
    return app;
}

const catalog = JSON.parse(readFileSync(new URL('./workflows/all-inspire-nodes.widgets.json', import.meta.url)));
const ADDED_BY_THE_PACK = ['Add to prompt', 'Remove Key', 'Save Settings', 'Refresh', 'Clear'];

export function makeNode(type, values = {}) {
    const widgets = catalog.nodes.find(node => node.type === type).widgets
        .filter(({ name }) => !ADDED_BY_THE_PACK.includes(name))
        .map(({ name, value }) => ({ name, value: name in values ? values[name] : value, options: {}, inputEl: {} }));
    return {
        comfyClass: type, type, widgets,
        addWidget(type, name, value, callback) {
            const widget = { type, name, value, callback };
            widgets.push(widget);
            return widget;
        },
    };
}
