import assert from 'node:assert/strict';
import test, { mock } from 'node:test';

import { makeApi, makeApp, makeNode, openPage } from './page.mjs';

function numbered(node) {
    for (const widget of node.widgets)
        if (widget.name === 'seed' || widget.name === 'noise_seed')
            widget.type = 'number';
    return node;
}

async function page(nodes, queued = []) {
    const app = makeApp({ _nodes_by_id: nodes });
    const api = makeApi({ queuePrompt: async (...args) => { queued.push(args); return { prompt_id: 'unit' }; } });
    const impact = { get_wildcards_list: () => [], get_wildcard_label: () => 'Select Wildcard', is_wildcard_label: () => false, load_wildcard_status: async () => {} };
    await openPage({ '/scripts/app.js': { app, ComfyApp: {} }, '/scripts/api.js': { api }, '/ComfyUI-Impact-Pack/impact-pack.js': impact },
                   'prompt.js', 'seed.js');
    return { app, api, queued };
}

function graph() {
    const seed = makeNode('GlobalSeed //Inspire', { value: 100, last_seed: '7' });
    seed.type = 'GlobalSeed //Inspire';
    const nodes = {
        1: seed,
        3: numbered(makeNode('KSampler //Inspire', { seed: 11 })),
        4: numbered(makeNode('SeedExplorer //Inspire')),
        5: numbered(makeNode('LoraLoaderBlockWeight //Inspire', { seed: 13 })),
    };
    for (const id in nodes)
        nodes[id].id = Number(id);
    return nodes;
}

const value = (node, name) => node.widgets.find(widget => widget.name === name).value;

test('the seeds the server sent are written into the seed widgets and the Global Seed', async () => {
    const nodes = graph();
    const { api } = await page(nodes);
    api.dispatch('inspire-global-seed', { value: 101, seed_map: { 3: 101 } });
    assert.equal(value(nodes[1], 'value'), 101);
    assert.equal(value(nodes[1], 'last_seed'), 100);
    assert.equal(value(nodes[3], 'seed'), 101);
    assert.equal(value(nodes[5], 'seed'), 13);
});

test('without a value from the server the Global Seed keeps its value and remembers it as the last seed', async () => {
    const nodes = graph();
    const { api } = await page(nodes);
    api.dispatch('inspire-global-seed', { value: null, seed_map: {} });
    assert.equal(value(nodes[1], 'value'), 100);
    assert.equal(value(nodes[1], 'last_seed'), 100);
});

test('the prompt goes to the server with the widget indices of every seed, sampler and scheduler', async () => {
    const nodes = graph();
    const { api, queued } = await page(nodes);
    const output = { 3: { class_type: 'KSampler //Inspire' } };
    await api.queuePrompt(0, { output, workflow: {} }, 'unit');
    assert.equal(queued.length, 1);
    const [number, { workflow }, extra] = queued[0];
    assert.equal(number, 0);
    assert.equal(extra, 'unit');
    assert.deepEqual(workflow.widget_idx_map, {
        3: { seed: 0, sampler_name: 4, scheduler: 5 },
        5: { seed: 5 },
    });
});

test('a seed widget converted to an input is left out of the indices', async () => {
    const nodes = graph();
    nodes[3].widgets[0].type = 'converted-widget';
    const { api, queued } = await page(nodes);
    await api.queuePrompt(0, { output: {}, workflow: {} });
    assert.deepEqual(queued[0][1].workflow.widget_idx_map[3], { sampler_name: 4, scheduler: 5 });
});

test('with a Global Seed in the prompt the next prompt waits for the seeds of the server', async () => {
    const { api } = await page(graph());
    let done = false;
    const queued = api.queuePrompt(0, { output: { 1: { class_type: 'GlobalSeed //Inspire' } }, workflow: {} }).then(() => { done = true; });
    for (let turn = 0; turn < 5; turn++)
        await Promise.resolve();
    assert.equal(done, false);
    api.dispatch('inspire-global-seed', { value: 101, seed_map: {} });
    await queued;
    assert.equal(done, true);
});

test('without a Global Seed the prompt does not wait', async () => {
    const { api } = await page(graph());
    const result = await api.queuePrompt(0, { output: { 3: { class_type: 'KSampler //Inspire' } }, workflow: {} });
    assert.deepEqual(result, { prompt_id: 'unit' });
});

test('when the seeds do not come the next prompt goes after one second', async () => {
    const { api } = await page(graph());
    mock.timers.enable({ apis: ['setTimeout'] });
    try {
        let done = false;
        const queued = api.queuePrompt(0, { output: { 1: { class_type: 'GlobalSeed //Inspire' } }, workflow: {} }).then(() => { done = true; });
        for (let turn = 0; turn < 5; turn++)
            await Promise.resolve();
        mock.timers.tick(999);
        await Promise.resolve();
        assert.equal(done, false);
        mock.timers.tick(1);
        await queued;
        assert.equal(done, true);
    }
    finally {
        mock.timers.reset();
    }
});
