import assert from 'node:assert/strict';
import test from 'node:test';
import {
  add, addBias, concatCols, crossEntropy, embedding, layerNorm, matmul, matmulBT, relu, scale, sliceCols, softmax,
} from '../src/autograd/ops.js';
import { Tensor } from '../src/autograd/tensor.js';
import { gradCheck, randomTensor, weightedSum } from './helpers.js';

const TOL = 1e-5;

test('matmul forward value and gradient', () => {
  const a = Tensor.from([1, 2, 3, 4], 2, 2);
  const b = Tensor.from([5, 6, 7, 8], 2, 2);
  assert.deepEqual(Array.from(matmul(a, b).data), [19, 22, 43, 50]);
  const A = randomTensor(3, 4, 1);
  const B = randomTensor(4, 5, 2);
  assert.ok(gradCheck(() => weightedSum(matmul(A, B)), [A, B]) < TOL);
});

test('matmulBT equals matmul with a transposed matrix, and has correct gradient', () => {
  const A = randomTensor(3, 4, 3);
  const B = randomTensor(5, 4, 4);
  const Bt = Tensor.zeros(4, 5);
  for (let i = 0; i < 5; i++) for (let j = 0; j < 4; j++) Bt.data[j * 5 + i] = B.data[i * 4 + j];
  const x = matmulBT(A, B).data;
  const y = matmul(A, Bt).data;
  x.forEach((v, i) => assert.ok(Math.abs(v - y[i]) < 1e-12));
  assert.ok(gradCheck(() => weightedSum(matmulBT(A, B)), [A, B]) < TOL);
});

test('add, addBias, scale, relu gradients', () => {
  const A = randomTensor(3, 4, 5);
  const B = randomTensor(3, 4, 6);
  const bias = randomTensor(1, 4, 7);
  assert.ok(gradCheck(() => weightedSum(add(A, B)), [A, B]) < TOL);
  assert.ok(gradCheck(() => weightedSum(addBias(A, bias)), [A, bias]) < TOL);
  assert.ok(gradCheck(() => weightedSum(scale(A, 0.37)), [A]) < TOL);
  assert.ok(gradCheck(() => weightedSum(relu(A)), [A]) < TOL);
});

test('softmax rows sum to 1, respect the mask, and have correct gradient', () => {
  const A = randomTensor(4, 5, 8);
  const mask = new Uint8Array(20).fill(1);
  mask[1] = 0; // row 0, col 1 blocked
  mask[7] = 0;
  const s = softmax(A, mask);
  for (let i = 0; i < 4; i++) {
    let total = 0;
    for (let j = 0; j < 5; j++) total += s.data[i * 5 + j];
    assert.ok(Math.abs(total - 1) < 1e-12);
  }
  assert.equal(s.data[1], 0);
  assert.equal(s.data[7], 0);
  assert.ok(gradCheck(() => weightedSum(softmax(A, mask)), [A]) < TOL);
});

test('softmax of a fully masked row is all zeros (no NaN)', () => {
  const A = randomTensor(2, 3, 9);
  const mask = new Uint8Array([0, 0, 0, 1, 1, 1]);
  const s = softmax(A, mask);
  assert.deepEqual(Array.from(s.data.slice(0, 3)), [0, 0, 0]);
  assert.ok(!Array.from(s.data).some(Number.isNaN));
});

test('layerNorm gives zero mean / unit variance per row, and correct gradient', () => {
  const X = randomTensor(3, 8, 10, 3);
  const gamma = randomTensor(1, 8, 11);
  const beta = randomTensor(1, 8, 12);
  const ones = Tensor.zeros(1, 8).data.fill(1);
  const zero = Tensor.zeros(1, 8);
  const out = layerNorm(X, new Tensor(ones, [1, 8]), zero);
  for (let i = 0; i < 3; i++) {
    const row = Array.from(out.data.slice(i * 8, i * 8 + 8));
    const mean = row.reduce((s, v) => s + v, 0) / 8;
    const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / 8;
    assert.ok(Math.abs(mean) < 1e-9);
    assert.ok(Math.abs(variance - 1) < 1e-3);
  }
  assert.ok(gradCheck(() => weightedSum(layerNorm(X, gamma, beta)), [X, gamma, beta]) < 1e-4);
});

test('embedding lookup, repeated ids accumulate gradient', () => {
  const table = randomTensor(6, 4, 13);
  const ids = [2, 5, 2, 0];
  const out = embedding(table, ids);
  assert.deepEqual(Array.from(out.data.slice(0, 4)), Array.from(table.data.slice(8, 12)));
  assert.ok(gradCheck(() => weightedSum(embedding(table, ids)), [table]) < TOL);
  assert.throws(() => embedding(table, [6]));
});

test('sliceCols and concatCols round-trip with correct gradients', () => {
  const A = randomTensor(3, 6, 14);
  const rebuilt = concatCols([sliceCols(A, 0, 2), sliceCols(A, 2, 6)]);
  assert.deepEqual(Array.from(rebuilt.data), Array.from(A.data));
  assert.ok(gradCheck(() => weightedSum(concatCols([sliceCols(A, 0, 2), sliceCols(A, 2, 6)])), [A]) < TOL);
});

test('crossEntropy matches -log(softmax) and ignores masked targets', () => {
  const L = randomTensor(3, 5, 15);
  const targets = [1, 4, -100];
  const p = softmax(L).data;
  const expected = (-Math.log(p[1]) - Math.log(p[5 + 4])) / 2;
  assert.ok(Math.abs(crossEntropy(L, targets).item() - expected) < 1e-12);
  assert.ok(gradCheck(() => crossEntropy(L, targets), [L]) < TOL);
});
