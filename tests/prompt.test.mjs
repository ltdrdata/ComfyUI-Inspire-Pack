import assert from 'node:assert/strict';
import test from 'node:test';

import { makeApi, makeApp, makeNode, openPage } from './page.mjs';

const LORA = 'Select the LoRA to add to the text';
const WILDCARD = 'Select Wildcard 🟢 Full Cache';

async function page({ impact = true, presets = {} } = {}) {
    const app = makeApp();
    const fetched = [];
    const api = makeApi({ fetchApi: async url => {
        fetched.push(url);
        return { json: async () => ({ presets: presets[new URL(url, 'http://page').searchParams.get('category')] ?? [] }) };
    } });
    const status = { loads: 0 };
    const frontend = { '/scripts/app.js': { app, ComfyApp: {} }, '/scripts/api.js': { api } };
    if (impact)
        frontend['/ComfyUI-Impact-Pack/impact-pack.js'] = {
            get_wildcards_list: () => [WILDCARD, '__animal__', '__color__'],
            get_wildcard_label: () => WILDCARD,
            is_wildcard_label: value => value === WILDCARD,
            load_wildcard_status: async () => { status.loads += 1; },
        };
    await openPage(frontend, 'prompt.js');
    const [extension] = app.extensions;
    const create = (type, values) => {
        const node = makeNode(type, values);
        extension.nodeCreated(node, app);
        return node;
    };
    return { app, create, fetched, status };
}

const widget = (node, name) => node.widgets.find(w => w.name === name);

async function pick(node, name, item, value = item) {
    const combo = widget(node, name);
    combo.value = item;
    await combo.callback(value === 'label' ? combo.value : value, {}, node);
}

for (const renderer of ['canvas', 'nodes 2.0']) {
    const value = renderer === 'canvas' ? 'label' : undefined;

    test(`wildcard encode: a picked lora is added to the wildcard text (${renderer})`, async () => {
        const { create } = await page();
        const node = create('WildcardEncode //Inspire', { wildcard_text: 'a fox' });
        await pick(node, 'Select to add LoRA', 'style/pixel.safetensors', value);
        await pick(node, 'Select to add LoRA', 'detail', value);
        assert.equal(widget(node, 'wildcard_text').value, 'a fox<lora:style/pixel><lora:detail>');
    });

    test(`wildcard encode: a picked wildcard is added after a comma (${renderer})`, async () => {
        const { app, create, status } = await page();
        const node = create('WildcardEncode //Inspire', { wildcard_text: '' });
        await pick(node, 'Select to add Wildcard', '__animal__', value);
        await pick(node, 'Select to add Wildcard', '__color__', value);
        assert.equal(widget(node, 'wildcard_text').value, '__animal__, __color__');
        assert.equal(status.loads, 2);
        assert.equal(app.canvas.dirty, 2);
    });

    test(`make basic pipe: picks go to the text that "Add selection to" names (${renderer})`, async () => {
        const { create } = await page();
        const node = create('MakeBasicPipe //Inspire', { positive_wildcard_text: 'a fox', negative_wildcard_text: 'blurry' });
        await pick(node, 'Select to add LoRA', 'pixel.safetensors', value);
        await pick(node, 'Select to add Wildcard', '__animal__', value);
        widget(node, 'Add selection to').value = false;
        await pick(node, 'Select to add LoRA', 'bad.safetensors', value);
        await pick(node, 'Select to add Wildcard', '__color__', value);
        assert.equal(widget(node, 'positive_wildcard_text').value, 'a fox<lora:pixel>, __animal__');
        assert.equal(widget(node, 'negative_wildcard_text').value, 'blurry<lora:bad>, __color__');
    });
}

test('the combos show their label and save it, whatever was picked', async () => {
    const { create } = await page();
    for (const type of ['WildcardEncode //Inspire', 'MakeBasicPipe //Inspire']) {
        const node = create(type);
        const lora = widget(node, 'Select to add LoRA');
        const wildcard = widget(node, 'Select to add Wildcard');
        lora.value = 'pixel.safetensors';
        wildcard.value = '__animal__';
        assert.equal(lora.value, LORA);
        assert.equal(wildcard.value, WILDCARD);
        assert.equal(lora.serializeValue(), LORA);
        assert.equal(wildcard.serializeValue(), 'Select the Wildcard to add to the text');
        assert.deepEqual(wildcard.options.values, [WILDCARD, '__animal__', '__color__']);
    }
});

test('without the Impact Pack the wildcard combo says so', async () => {
    const { create } = await page({ impact: false });
    const node = create('WildcardEncode //Inspire');
    assert.deepEqual(widget(node, 'Select to add Wildcard').options.values, ["Impact Pack isn't installed or needs browser cache refresh."]);
    assert.equal(widget(node, 'Select to add Wildcard').value, 'Select the Wildcard to add to the text');
});

test('the mode decides whether the populated text can be edited, and an old boolean mode reads as populate or fixed', async () => {
    const { create } = await page();
    const node = create('WildcardEncode //Inspire');
    const mode = widget(node, 'mode');
    const populated = widget(node, 'populated_text');
    assert.equal(mode.value, 'populate');
    for (const [given, read, locked] of [[false, 'fixed', false], [true, 'populate', true], ['reproduce', 'reproduce', false], ['populate', 'populate', true]]) {
        mode.value = given;
        assert.equal(mode.value, read);
        assert.equal(populated.inputEl.disabled, locked);
    }
    const pipe = create('MakeBasicPipe //Inspire');
    widget(pipe, 'wildcard_mode').value = false;
    assert.equal(widget(pipe, 'wildcard_mode').value, 'fixed');
    assert.equal(widget(pipe, 'positive_populated_text').inputEl.disabled, false);
    assert.equal(widget(pipe, 'negative_populated_text').inputEl.disabled, false);
});

test('prompt builder: the presets of the category come from the server and a picked preset is added to the text', async () => {
    const { create, fetched } = await page({ presets: { 'Angle of View': ['Close-Up', 'camera:From Above'] } });
    const node = create('PromptBuilder //Inspire', { category: 'Angle of View', text: '' });
    const preset = widget(node, 'preset');
    assert.deepEqual(preset.options.values, ['#PRESET']);
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(fetched, ['/inspire/prompt_builder?category=Angle of View']);
    assert.deepEqual(preset.options.values, ['Close-Up', 'camera:From Above']);
    assert.equal(fetched.length, 1);
    await pick(node, 'preset', 'Close-Up', 'label');
    await pick(node, 'preset', 'camera:From Above');
    assert.equal(widget(node, 'text').value, 'Close-Up, From Above');
    assert.equal(preset.value, '#PRESET');
    assert.equal(preset.serializeValue(), '#PRESET');
});

test('seed explorer: the button adds the additional seed with its strength', async () => {
    const { create } = await page();
    const node = create('SeedExplorer //Inspire', { seed_prompt: '', additional_seed: 5, additional_strength: 0.3 });
    const button = widget(node, 'Add to prompt');
    button.callback();
    assert.equal(widget(node, 'seed_prompt').value, '5');
    widget(node, 'additional_seed').value = 8;
    button.callback();
    button.callback();
    assert.equal(widget(node, 'seed_prompt').value, '5, 8:0.30, 9:0.30');
    assert.equal(widget(node, 'additional_seed').value, 10);
});

test('regional seed explorer: the button never makes the additional seed the initial seed', async () => {
    const { create } = await page();
    const node = create('RegionalSeedExplorerMask //Inspire', { seed_prompt: '', additional_seed: 5, additional_strength: 0.25 });
    const button = widget(node, 'Add to prompt');
    button.callback();
    button.callback();
    assert.equal(widget(node, 'seed_prompt').value, '5:0.25, 6:0.25');
});
