import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { makeApi, makeApp, makeNode, openPage } from './page.mjs';

const PRESETS = readFileSync(new URL('../resources/lbw-preset.txt', import.meta.url), 'utf8').split('\n').map(line => line.trim());
const preset = name => PRESETS.find(line => line.startsWith(name + ':'));

async function create(type, values) {
    const app = makeApp();
    await openPage({ '/scripts/app.js': { app, ComfyApp: {} }, '/scripts/api.js': { api: makeApi() } }, 'lora_block_weight.js');
    const node = makeNode(type, values);
    node.widgets.find(w => w.name === 'lora_name').options.values = ['sd15/style.safetensors', 'sd15/detail.safetensors', 'xl/pixel.safetensors'];
    app.extensions[0].nodeCreated(node, app);
    return node;
}

const widget = (node, name) => node.widgets.find(w => w.name === name);

function pick(node, item) {
    const combo = widget(node, 'preset');
    combo.value = item;
    combo.callback(combo.value, {}, node);
}

for (const type of ['LoraLoaderBlockWeight //Inspire', 'MakeLBW //Inspire']) {
    test(`${type}: a picked preset puts its vector into block_vector`, async () => {
        const node = await create(type, { block_vector: '1,1,1' });
        pick(node, preset('SD-INS'));
        assert.equal(widget(node, 'block_vector').value, '1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0');
        assert.equal(widget(node, 'preset').value, preset('SD-INS'));
        pick(node, preset('SDXL-OUTALL'));
        assert.equal(widget(node, 'block_vector').value, '1,0,0,0,0,0,1,1,1,1,1,1');
    });

    test(`${type}: the lora list shows the loras of the chosen folder`, async () => {
        const node = await create(type);
        const loras = widget(node, 'lora_name').options;
        assert.deepEqual(loras.values, ['sd15/style.safetensors', 'sd15/detail.safetensors', 'xl/pixel.safetensors']);
        widget(node, 'category_filter').value = 'sd15';
        assert.deepEqual(loras.values, ['sd15/style.safetensors', 'sd15/detail.safetensors']);
        loras.values = ['sd15/new.safetensors', 'xl/new.safetensors'];
        assert.deepEqual(loras.values, ['sd15/new.safetensors']);
    });
}

test('XY input: a picked preset adds a line of the vector and the preset name as its label', async () => {
    const node = await create('XY Input: Lora Block Weight //Inspire', { block_vectors: '' });
    pick(node, preset('SD-ALL'));
    pick(node, preset('SD-NONE'));
    assert.equal(widget(node, 'block_vectors').value, `${preset('SD-ALL')}/SD-ALL\n${preset('SD-NONE')}/SD-NONE`);
});

test('XY input: a test preset writes one line for each block with A in that block', async () => {
    const node = await create('XY Input: Lora Block Weight //Inspire', { block_vectors: 'kept' });
    pick(node, preset('@SD-FULL-TEST'));
    const lines = widget(node, 'block_vectors').value.trimEnd().split('\n');
    assert.equal(lines.length, 17);
    assert.equal(lines[0], 'B1:A,' + Array(16).fill('0').join(','));
    assert.equal(lines[16], 'B17:' + Array(16).fill('0').join(',') + ',A');
});

test('XY input: a test preset for a block writes one line for each of its parts', async () => {
    const node = await create('XY Input: Lora Block Weight //Inspire', { block_vectors: '' });
    pick(node, preset('@SD-BLOCK3-TEST'));
    const lines = widget(node, 'block_vectors').value.trimEnd().split('\n');
    assert.equal(lines.length, 12);
    const others = Array(12).fill('0 ');
    assert.equal(lines[0], `B3.SUB1:0,0, A ${others.slice(1).join('')},` + Array(14).fill('0').join(','));
    assert.equal(lines[11], `B3.SUB12:0,0, ${others.slice(1).join('')}A ,` + Array(14).fill('0').join(','));
});
