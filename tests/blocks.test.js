import assert from 'node:assert/strict';
import test from 'node:test';
import { MultiHeadAttention } from '../src/attention/multi_head.js';
import { scaledDotProductAttention } from '../src/attention/scaled_dot_product.js';
import { Embedding } from '../src/embeddings/embedding.js';
import { causalMask, combineMasks, paddingMask } from '../src/masks/masks.js';
import { PositionalEncoding } from '../src/positional/sinusoidal.js';
import { AddNorm } from '../src/addnorm/add_norm.js';
import { FeedForward } from '../src/feedforward/feed_forward.js';
import { EncoderLayer } from '../src/encoder/encoder_layer.js';
import { DecoderLayer } from '../src/decoder/decoder_layer.js';
import { mulberry32 } from '../src/utils/rng.js';
import { gradCheck, randomTensor, weightedSum } from './helpers.js';

const cfg = { dModel: 8, heads: 2, dFF: 16 };

test('embedding: output shape and sqrt(d) scale', () => {
  const e = new Embedding(10, 8, mulberry32(1));
  const out = e.forward([3, 4, 3]);
  assert.deepEqual(out.shape, [3, 8]);
  assert.ok(Math.abs(out.data[0] - e.table.data[3 * 8] * Math.sqrt(8)) < 1e-12);
});

test('positional encoding: known values and shape preserved', () => {
  const pe = new PositionalEncoding(8, 16);
  // position 0 -> sin(0)=0, cos(0)=1 alternating
  assert.deepEqual(Array.from(pe.table.slice(0, 4)), [0, 1, 0, 1]);
  // position 1, first pair uses angle 1
  assert.ok(Math.abs(pe.table[8] - Math.sin(1)) < 1e-12);
  assert.ok(Math.abs(pe.table[9] - Math.cos(1)) < 1e-12);
  const x = randomTensor(5, 8, 1);
  assert.deepEqual(pe.forward(x).shape, [5, 8]);
  assert.throws(() => pe.forward(randomTensor(17, 8, 1)));
});

test('masks: causal, padding and combined', () => {
  assert.deepEqual(Array.from(causalMask(3)), [1, 0, 0, 1, 1, 0, 1, 1, 1]);
  assert.deepEqual(Array.from(paddingMask([5, 0, 6], 2)), [1, 0, 1, 1, 0, 1]);
  assert.deepEqual(Array.from(combineMasks(causalMask(3), paddingMask([5, 0, 6], 3))), [1, 0, 0, 1, 0, 0, 1, 0, 1]);
});

test('scaled dot-product attention: weights are a distribution, masked keys get 0', () => {
  const Q = randomTensor(3, 4, 1);
  const K = randomTensor(3, 4, 2);
  const V = randomTensor(3, 4, 3);
  const { out, weights } = scaledDotProductAttention(Q, K, V, causalMask(3));
  assert.deepEqual(out.shape, [3, 4]);
  assert.equal(weights.data[1], 0);
  assert.ok(Math.abs(weights.data[0] - 1) < 1e-12); // first position can only see itself
});

test('multi-head attention: shape, per-head maps, gradient', () => {
  const mha = new MultiHeadAttention(8, 2, mulberry32(4));
  const x = randomTensor(5, 8, 5);
  const m = randomTensor(4, 8, 6);
  const out = mha.forward(x, m);
  assert.deepEqual(out.shape, [5, 8]);
  assert.equal(mha.lastWeights.length, 2);
  assert.deepEqual(mha.lastWeights[0].shape, [5, 4]);
  assert.ok(gradCheck(() => weightedSum(mha.forward(x, m)), [x, m, mha.Wq, mha.Wo]) < 1e-5);
  assert.throws(() => new MultiHeadAttention(10, 3, mulberry32(1)));
});

test('add & norm and feed forward: shapes and gradients', () => {
  const an = new AddNorm(8);
  const ff = new FeedForward(8, 16, mulberry32(7));
  const x = randomTensor(4, 8, 8);
  assert.deepEqual(an.forward(x, ff.forward(x)).shape, [4, 8]);
  assert.ok(gradCheck(() => weightedSum(an.forward(x, ff.forward(x))), [x, ff.W1, ff.b1, an.gamma]) < 1e-4);
});

test('encoder layer and decoder layer: shapes', () => {
  const enc = new EncoderLayer(cfg, mulberry32(9));
  const dec = new DecoderLayer(cfg, mulberry32(10));
  const src = randomTensor(5, 8, 11);
  const tgt = randomTensor(3, 8, 12);
  const mem = enc.forward(src, paddingMask([1, 1, 1, 1, 1], 5));
  assert.deepEqual(mem.shape, [5, 8]);
  const out = dec.forward(tgt, mem, causalMask(3), paddingMask([1, 1, 1, 1, 1], 3));
  assert.deepEqual(out.shape, [3, 8]);
});

test('attention to a padded key position is exactly zero', () => {
  const mha = new MultiHeadAttention(8, 2, mulberry32(13));
  const x = randomTensor(4, 8, 14);
  mha.forward(x, x, paddingMask([5, 6, 0, 0], 4));
  for (const w of mha.lastWeights) {
    for (let i = 0; i < 4; i++) {
      assert.equal(w.data[i * 4 + 2], 0);
      assert.equal(w.data[i * 4 + 3], 0);
    }
  }
});
