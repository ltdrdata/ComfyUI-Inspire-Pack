import assert from 'node:assert/strict';
import test from 'node:test';

import { makeApi, makeApp, makeNode, openPage } from './page.mjs';

async function extension() {
    const app = makeApp();
    await openPage({ '/scripts/app.js': { app, ComfyApp: {} }, '/scripts/api.js': { api: makeApi() } }, 'inspire-pack.js');
    return { app, extension: app.extensions[0] };
}

function withOutputs(node, outputs) {
    node.outputs = outputs;
    node.addOutput = (name, type) => node.outputs.push({ name, type });
    node.removeOutput = index => node.outputs.splice(index, 1);
    return node;
}

async function splitter(type, outputType, count) {
    const { app, extension: inspire } = await extension();
    const node = withOutputs(makeNode(type, { split_count: count }), [{ name: outputType.toLowerCase(), type: outputType, label: outputType }]);
    inspire.nodeCreated(node, app);
    return node;
}

const outputs = node => node.outputs.map(output => [output.name, output.type]);

test('image batch splitter: one output for each picture of split_count and one for the rest', async () => {
    const node = await splitter('ImageBatchSplitter //Inspire', 'IMAGE', 3);
    assert.deepEqual(outputs(node), [['image 1', 'IMAGE'], ['image 2', 'IMAGE'], ['image 3', 'IMAGE'], ['remained', 'IMAGE']]);
    assert.equal(node.outputs[0].label, undefined);
    node.widgets[0].value = 1;
    assert.deepEqual(outputs(node), [['image 1', 'IMAGE'], ['remained', 'IMAGE']]);
    assert.equal(node.widgets[0].value, 1);
    node.widgets[0].value = 51;
    node.widgets[0].value = -1;
    assert.equal(node.widgets[0].value, 1);
});

test('latent batch splitter: its outputs are latents', async () => {
    const node = await splitter('LatentBatchSplitter //Inspire', 'LATENT', 2);
    assert.deepEqual(outputs(node), [['latent 1', 'LATENT'], ['latent 2', 'LATENT'], ['remained', 'LATENT']]);
    node.widgets[0].value = 0;
    assert.deepEqual(outputs(node), [['remained', 'LATENT']]);
});

const CONCAT = 'ConcatConditioningsWithMultiplier //Inspire';

async function concat(widgetsValues) {
    const { app, extension: inspire } = await extension();
    class NodeType {}
    await inspire.beforeRegisterNodeDef(NodeType, { name: CONCAT }, app);
    const node = Object.assign(new NodeType(), makeNode(CONCAT), {
        inputs: [{ name: 'conditioning1', type: 'CONDITIONING', link: null }],
        outputs: [{ name: 'CONDITIONING', type: 'CONDITIONING', links: [] }],
        widgets_values: widgetsValues,
        addInput(name, type) { this.inputs.push({ name, type, link: null }); },
        removeInput(index) { this.inputs.splice(Number(index), 1); },
        disconnectInput(index) { this.inputs[index].link = null; },
        computeSize() { return [1, 1]; },
        setSize() {},
    });
    return node;
}

async function connect(node, index, link) {
    node.inputs[index].link = link;
    await node.onConnectionsChange(1, index, link !== null, { type: 'CONDITIONING' });
}

const shape = node => ({
    inputs: node.inputs.map(input => [input.name, input.link]),
    multipliers: node.widgets.map(widget => [widget.name, widget.value]),
});

test('concat conditionings: each linked conditioning gets a multiplier and a free input follows', async () => {
    const node = await concat();
    await connect(node, 0, 11);
    assert.deepEqual(shape(node), { inputs: [['conditioning1', 11], ['conditioning2', null]], multipliers: [['multiplier1', 1]] });
    await connect(node, 1, 12);
    assert.deepEqual(shape(node), {
        inputs: [['conditioning1', 11], ['conditioning2', 12], ['conditioning3', null]],
        multipliers: [['multiplier1', 1], ['multiplier2', 1]],
    });
});

test('concat conditionings: a removed link closes the gap and the next conditioning keeps its multiplier', async () => {
    const node = await concat();
    await connect(node, 0, 11);
    await connect(node, 1, 12);
    node.widgets[1].value = 0.5;
    await connect(node, 0, null);
    assert.deepEqual(shape(node), { inputs: [['conditioning1', 12], ['conditioning2', null]], multipliers: [['multiplier1', 0.5]] });
});

test('concat conditionings: a link of another type changes nothing', async () => {
    const node = await concat();
    node.inputs[0].link = 11;
    await node.onConnectionsChange(1, 0, true, { type: 'MODEL' });
    assert.deepEqual(shape(node), { inputs: [['conditioning1', 11]], multipliers: [['multiplier1', 1]] });
});

test('concat conditionings: a loaded workflow gets back a multiplier for each saved value', async () => {
    const node = await concat([1, 0.5, 2]);
    node.inputs.push({ name: 'conditioning2', type: 'CONDITIONING', link: 12 }, { name: 'conditioning3', type: 'CONDITIONING', link: 13 });
    async function loadGraphData() {
        node.inputs[0].link = 11;
        await node.onConnectionsChange(1, 0, true, { type: 'CONDITIONING' });
    }
    await loadGraphData();
    assert.deepEqual(shape(node).multipliers.map(([name]) => name), ['multiplier1', 'multiplier2', 'multiplier3']);
    assert.equal(node.inputs.length, 3);
});
