// One block of the right stack (the grey box marked Nx on the right):
//   Masked Multi-Head Attention -> Add & Norm
//   Multi-Head Attention over the encoder output -> Add & Norm
//   Feed Forward -> Add & Norm
import { AddNorm } from '../addnorm/add_norm.js';
import { MultiHeadAttention } from '../attention/multi_head.js';
import { FeedForward } from '../feedforward/feed_forward.js';
import { Module } from '../nn/module.js';

export class DecoderLayer extends Module {
  constructor(cfg, rng) {
    super();
    this.maskedSelfAttn = new MultiHeadAttention(cfg.dModel, cfg.heads, rng);
    this.norm1 = new AddNorm(cfg.dModel);
    this.crossAttn = new MultiHeadAttention(cfg.dModel, cfg.heads, rng);
    this.norm2 = new AddNorm(cfg.dModel);
    this.ffn = new FeedForward(cfg.dModel, cfg.dFF, rng);
    this.norm3 = new AddNorm(cfg.dModel);
  }

  /**
   * x         decoder state [nt, d]
   * memory    encoder output [ns, d]  (the arrow coming across from the left)
   * selfMask  causal (+ padding) mask [nt*nt]
   * crossMask padding mask over the source [nt*ns]
   */
  forward(x, memory, selfMask, crossMask) {
    x = this.norm1.forward(x, this.maskedSelfAttn.forward(x, x, selfMask));
    x = this.norm2.forward(x, this.crossAttn.forward(x, memory, crossMask));
    return this.norm3.forward(x, this.ffn.forward(x));
  }
}
