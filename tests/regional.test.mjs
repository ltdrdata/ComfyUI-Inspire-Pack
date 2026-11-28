import assert from 'node:assert/strict';
import test from 'node:test';

import { makeApi, makeApp, openPage } from './page.mjs';

async function applyNode(type, inputs) {
    const app = makeApp();
    await openPage({ '/scripts/app.js': { app, ComfyApp: {} }, '/scripts/api.js': { api: makeApi() } }, 'regional.js');
    class NodeType {}
    await app.extensions[0].beforeRegisterNodeDef(NodeType, { name: type }, app);
    return Object.assign(new NodeType(), {
        inputs: inputs.map(([name, type]) => ({ name, type, link: null })),
        addInput(name, type) { this.inputs.push({ name, type, link: null }); },
        removeInput(index) { this.inputs.splice(Number(index), 1); },
    });
}

function link(node, index, id) {
    node.inputs[index].link = id;
    node.onConnectionsChange(1, index, id !== null, { id });
}

const slots = node => node.inputs.map(input => [input.name, input.link]);

const adapters = [['model', 'MODEL'], ['positive', 'CONDITIONING'], ['negative', 'CONDITIONING'], ['regional_adapter1', 'REGIONAL_ADAPTER']];

test('apply regional adapters: each linked entry gets a free input after it, and an unlinked one goes away', async () => {
    const node = await applyNode('ApplyRegionalAdapters //Inspire', adapters);
    link(node, 0, 10);
    link(node, 3, 11);
    link(node, 4, 12);
    assert.deepEqual(slots(node), [['model', 10], ['positive', null], ['negative', null], ['regional_adapter1', 11], ['regional_adapter2', 12], ['regional_adapter3', null]]);
    assert.equal(node.inputs[5].type, 'REGIONAL_ADAPTER');
    link(node, 3, null);
    assert.deepEqual(slots(node), [['model', 10], ['positive', null], ['negative', null], ['regional_adapter1', 12], ['regional_adapter2', null]]);
});

test('apply regional adapters: unlinking the model or a prompt keeps its input and the entries', async () => {
    const node = await applyNode('ApplyRegionalAdapters //Inspire', adapters);
    [10, 20, 21, 11, 12].forEach((id, index) => link(node, index, id));
    [0, 1, 2].forEach(index => link(node, index, null));
    assert.deepEqual(slots(node), [['model', null], ['positive', null], ['negative', null], ['regional_adapter1', 11], ['regional_adapter2', 12], ['regional_adapter3', null]]);
});

test('ApplyRegionalIPAdapters //Inspire: unlinking ipadapter_pipe keeps its input and the entries', async () => {
    const node = await applyNode('ApplyRegionalIPAdapters //Inspire', [['ipadapter_pipe', 'IPADAPTER_PIPE'], ['regional_ipadapter1', 'REGIONAL_IPADAPTER']]);
    link(node, 0, 10);
    link(node, 1, 11);
    link(node, 2, 12);
    link(node, 0, null);
    assert.deepEqual(slots(node), [['ipadapter_pipe', null], ['regional_ipadapter1', 11], ['regional_ipadapter2', 12], ['regional_ipadapter3', null]]);
});
