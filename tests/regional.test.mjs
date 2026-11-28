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

test('apply regional adapters: each linked entry gets a free input after it, and an unlinked one goes away', async () => {
    const node = await applyNode('ApplyRegionalAdapters //Inspire', [['model', 'MODEL'], ['regional_adapter1', 'REGIONAL_ADAPTER']]);
    link(node, 0, 10);
    link(node, 1, 11);
    link(node, 2, 12);
    assert.deepEqual(slots(node), [['model', 10], ['regional_adapter1', 11], ['regional_adapter2', 12], ['regional_adapter3', null]]);
    assert.equal(node.inputs[3].type, 'REGIONAL_ADAPTER');
    link(node, 1, null);
    assert.deepEqual(slots(node), [['model', 10], ['regional_adapter1', 12], ['regional_adapter2', null]]);
});

for (const [type, first, entry] of [['ApplyRegionalAdapters //Inspire', ['model', 'MODEL'], ['regional_adapter', 'REGIONAL_ADAPTER']],
                                    ['ApplyRegionalIPAdapters //Inspire', ['ipadapter_pipe', 'IPADAPTER_PIPE'], ['regional_ipadapter', 'REGIONAL_IPADAPTER']]]) {
    test(`${type}: unlinking ${first[0]} keeps its input and the entries`, async () => {
        const node = await applyNode(type, [first, [`${entry[0]}1`, entry[1]]]);
        link(node, 0, 10);
        link(node, 1, 11);
        link(node, 2, 12);
        link(node, 0, null);
        assert.deepEqual(slots(node), [[first[0], null], [`${entry[0]}1`, 11], [`${entry[0]}2`, 12], [`${entry[0]}3`, null]]);
    });
}
