// The whole left side of the diagram:
//   Inputs -> Input Embedding -> + Positional Encoding -> Nx encoder layers
import { paddingMask } from '../masks/masks.js';
import { Module } from '../nn/module.js';
import { EncoderLayer } from './encoder_layer.js';

export class Encoder extends Module {
  constructor(cfg, embed, pos, rng) {
    super();
    this.embed = embed; // shared with the decoder and the output layer
    this.pos = pos;
    this.layers = Array.from({ length: cfg.layers }, () => new EncoderLayer(cfg, rng));
  }

  /** srcIds: array of token ids -> Tensor [n, dModel] ("memory" for the decoder) */
  forward(srcIds) {
    const mask = paddingMask(srcIds, srcIds.length);
    let x = this.pos.forward(this.embed.forward(srcIds));
    for (const layer of this.layers) x = layer.forward(x, mask);
    return x;
  }
}
