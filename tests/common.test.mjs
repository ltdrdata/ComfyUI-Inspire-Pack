import assert from 'node:assert/strict';
import test from 'node:test';

import { makeApi, makeApp, makeNode, openPage } from './page.mjs';

async function page(nodes) {
    const app = makeApp({ _nodes_by_id: nodes });
    const api = makeApi();
    await openPage({ '/scripts/app.js': { app, ComfyApp: {} }, '/scripts/api.js': { api } }, 'common.js');
    return api;
}

const widget = (node, name) => node.widgets.find(w => w.name === name);

test('feedback of the server sets the widget it names', async () => {
    const node = makeNode('KSampler //Inspire');
    const other = makeNode('KSampler //Inspire');
    const api = await page({ 3: node, 4: other });
    api.dispatch('inspire-node-feedback', { node_id: '3', widget_name: 'sampler_name', type: 'text', data: 'dpmpp_2m' });
    api.dispatch('inspire-node-feedback', { node_id: '9', widget_name: 'sampler_name', type: 'text', data: 'lms' });
    assert.equal(widget(node, 'sampler_name').value, 'dpmpp_2m');
    assert.equal(widget(other, 'sampler_name').value, 'euler');
});

test('an output label of the server names the output', async () => {
    const node = { outputs: [{ name: 'a' }, { name: 'b' }] };
    const api = await page({ 5: node });
    api.dispatch('inspire-node-output-label', { node_id: '5', output_idx: 1, label: 'unit label' });
    assert.deepEqual(node.outputs, [{ name: 'a' }, { name: 'b', label: 'unit label' }]);
});
