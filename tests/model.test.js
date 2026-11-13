import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { noGrad } from '../src/autograd/tensor.js';
import { CharTokenizer } from '../src/data/tokenizer.js';
import { generateIds, generateText } from '../src/generate/decode.js';
import { BOS, EOS } from '../src/masks/masks.js';
import { loadModel, saveModel } from '../src/model/checkpoint.js';
import { Transformer } from '../src/model/transformer.js';
import { Adam, clipGradNorm, lrSchedule } from '../src/optim/adam.js';
import { gradCheck } from './helpers.js';

const cfg = { vocabSize: 13, dModel: 16, heads: 2, layers: 2, dFF: 32, maxLen: 16 };

test('forward shapes and softmax output probabilities sum to 1', () => {
  const m = new Transformer(cfg, 1);
  const src = [3, 4, 5, 6];
  const tgtIn = [BOS, 6, 5];
  const logits = m.forward(src, tgtIn);
  assert.deepEqual(logits.shape, [3, 13]);
  const probs = m.head.probabilities(m.decode(tgtIn, m.encode(src), src));
  for (let i = 0; i < 3; i++) {
    let s = 0;
    for (let j = 0; j < 13; j++) s += probs.data[i * 13 + j];
    assert.ok(Math.abs(s - 1) < 1e-9);
  }
});

test('the embedding table is shared (tied) and counted once', () => {
  const m = new Transformer(cfg, 1);
  const names = m.parameters().map(([n]) => n);
  assert.equal(names.filter((n) => n.endsWith('table')).length, 1);
  assert.equal(names[0], 'embed.table');
  const total = m.countParameters();
  // embedding 13*16 + per encoder layer + per decoder layer
  const attn = 4 * 16 * 16;
  const ffn = 16 * 32 + 32 + 32 * 16 + 16;
  const norm = 2 * 16;
  const expected = 13 * 16 + 2 * (attn + ffn + 2 * norm) + 2 * (2 * attn + ffn + 3 * norm);
  assert.equal(total, expected);
});

test('decoder is causal: changing a later input token does not change earlier outputs', () => {
  const m = new Transformer(cfg, 2);
  const src = [3, 4, 5];
  const a = m.forward(src, [BOS, 4, 5, 6]);
  const b = m.forward(src, [BOS, 4, 9, 10]); // positions 2 and 3 changed
  for (let i = 0; i < 2 * 13; i++) assert.ok(Math.abs(a.data[i] - b.data[i]) < 1e-12, `row <2 changed at ${i}`);
  let changed = false;
  for (let i = 2 * 13; i < 3 * 13; i++) if (Math.abs(a.data[i] - b.data[i]) > 1e-9) changed = true;
  assert.ok(changed, 'later rows should change');
});

test('decoder output depends on the source (cross-attention is wired up)', () => {
  const m = new Transformer(cfg, 3);
  const a = m.forward([3, 4, 5], [BOS, 4]);
  const b = m.forward([6, 7, 8], [BOS, 4]);
  assert.ok(a.data.some((v, i) => Math.abs(v - b.data[i]) > 1e-6));
});

test('gradients of the whole model match finite differences', () => {
  const m = new Transformer({ ...cfg, layers: 1 }, 4);
  const src = [3, 4, 5, 6];
  const tgtIn = [BOS, 6, 5, 4, 3];
  const tgtOut = [6, 5, 4, 3, EOS];
  const params = m.parameters();
  const pick = (name) => params.find(([n]) => n === name)[1];
  const inputs = [
    pick('embed.table'),
    pick('encoder.layers.0.selfAttn.Wq'),
    pick('encoder.layers.0.ffn.W1'),
    pick('decoder.layers.0.maskedSelfAttn.Wk'),
    pick('decoder.layers.0.crossAttn.Wv'),
    pick('decoder.layers.0.norm3.gamma'),
  ];
  const worst = gradCheck(() => m.loss(src, tgtIn, tgtOut), inputs, { maxChecks: 8 });
  assert.ok(worst < 1e-4, `worst relative error ${worst}`);
});

test('a few optimiser steps reduce the loss on a fixed example', () => {
  const m = new Transformer(cfg, 5);
  const params = m.parameters();
  const adam = new Adam(params);
  const src = [3, 4, 5, 6];
  const tgtIn = [BOS, 6, 5, 4, 3];
  const tgtOut = [6, 5, 4, 3, EOS];
  const first = m.loss(src, tgtIn, tgtOut).item();
  for (let i = 0; i < 40; i++) {
    m.zeroGrad();
    m.loss(src, tgtIn, tgtOut).backward();
    clipGradNorm(params, 1);
    adam.step(0.01);
  }
  const last = m.loss(src, tgtIn, tgtOut).item();
  assert.ok(last < first * 0.2, `loss ${first} -> ${last}`);
});

test('lr schedule warms up then decays', () => {
  assert.ok(lrSchedule(10, 0.01, 100) < lrSchedule(100, 0.01, 100));
  assert.ok(lrSchedule(400, 0.01, 100) < lrSchedule(100, 0.01, 100));
  assert.ok(Math.abs(lrSchedule(100, 0.01, 100) - 0.01) < 1e-12);
});

test('save then load gives the same logits (float32 rounding only)', () => {
  const m = new Transformer(cfg, 6);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  saveModel(m, dir, { task: 'reverse', alphabet: '0123456789', stepsDone: 7 });
  const { model, config } = loadModel(dir);
  assert.equal(config.stepsDone, 7);
  const a = m.forward([3, 4], [BOS, 4]);
  const b = model.forward([3, 4], [BOS, 4]);
  a.data.forEach((v, i) => assert.ok(Math.abs(v - b.data[i]) < 1e-4));
  fs.rmSync(dir, { recursive: true });
});

test('tokenizer round trip and errors', () => {
  const tok = new CharTokenizer('0123456789');
  assert.equal(tok.vocabSize, 13);
  assert.deepEqual(tok.encode('907'), [12, 3, 10]);
  assert.equal(tok.decode([12, 3, 10, EOS, 5]), '907');
  assert.throws(() => tok.encode('a'));
});

test('generation terminates, respects maxNew, and is repeatable when greedy', () => {
  const m = new Transformer(cfg, 7);
  const tok = new CharTokenizer('0123456789');
  const one = generateIds(m, tok.encode('123'), { maxNew: 5 });
  assert.ok(one.ids.length <= 5);
  assert.equal(generateText(m, tok, '123', { maxNew: 5 }), generateText(m, tok, '123', { maxNew: 5 }));
  noGrad(() => assert.ok(m.encode([3, 4]).parents.length === 0)); // no graph is built in noGrad mode
});
