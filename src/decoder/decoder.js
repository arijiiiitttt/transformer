// The whole right side of the diagram, up to (not including) Linear:
//   Outputs (shifted right) -> Output Embedding -> + Positional Encoding -> Nx decoder layers
import { causalMask, combineMasks, paddingMask } from '../masks/masks.js';
import { Module } from '../nn/module.js';
import { DecoderLayer } from './decoder_layer.js';

export class Decoder extends Module {
  constructor(cfg, embed, pos, rng) {
    super();
    this.embed = embed;
    this.pos = pos;
    this.layers = Array.from({ length: cfg.layers }, () => new DecoderLayer(cfg, rng));
  }

  /**
   * tgtIds  tokens fed in so far, starting with BOS ("shifted right")
   * memory  encoder output
   * srcIds  source ids (so padding in the source can be masked)
   */
  forward(tgtIds, memory, srcIds) {
    const nt = tgtIds.length;
    const selfMask = combineMasks(causalMask(nt), paddingMask(tgtIds, nt));
    const crossMask = paddingMask(srcIds, nt);
    let x = this.pos.forward(this.embed.forward(tgtIds));
    for (const layer of this.layers) x = layer.forward(x, memory, selfMask, crossMask);
    return x;
  }
}
