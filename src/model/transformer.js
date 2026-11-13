// The full Transformer: every box of the diagram wired together.
//
//   srcIds --> [Input Embedding + Positional Encoding] --> Encoder (Nx) --+
//                                                                         | memory
//   BOS+tgt -> [Output Embedding + Positional Encoding] -> Decoder (Nx) <-+
//                                                              |
//                                                  Linear -> Softmax -> probabilities
import { crossEntropy } from '../autograd/ops.js';
import { Decoder } from '../decoder/decoder.js';
import { Embedding } from '../embeddings/embedding.js';
import { Encoder } from '../encoder/encoder.js';
import { Module } from '../nn/module.js';
import { OutputHead } from '../output/output_head.js';
import { PositionalEncoding } from '../positional/sinusoidal.js';
import { mulberry32 } from '../utils/rng.js';

export const DEFAULT_CONFIG = {
  vocabSize: 13,
  dModel: 64,
  heads: 4,
  layers: 2,
  dFF: 128,
  maxLen: 32,
};

export class Transformer extends Module {
  constructor(config = {}, seed = 1) {
    super();
    const cfg = { ...DEFAULT_CONFIG, ...config };
    this.cfg = cfg;
    const rng = mulberry32(seed);
    // order matters for parameter names: the shared table is registered first
    this.embed = new Embedding(cfg.vocabSize, cfg.dModel, rng);
    this.pos = new PositionalEncoding(cfg.dModel, cfg.maxLen);
    this.encoder = new Encoder(cfg, this.embed, this.pos, rng);
    this.decoder = new Decoder(cfg, this.embed, this.pos, rng);
    this.head = new OutputHead(this.embed);
  }

  encode(srcIds) {
    return this.encoder.forward(srcIds);
  }

  decode(tgtIds, memory, srcIds) {
    return this.decoder.forward(tgtIds, memory, srcIds);
  }

  /** logits [tgtIds.length, vocab] for teacher-forced decoder input tgtIds */
  forward(srcIds, tgtIds) {
    return this.head.logits(this.decode(tgtIds, this.encode(srcIds), srcIds));
  }

  /** training loss for one (source, decoder-input, expected-output) triple */
  loss(srcIds, tgtIn, tgtOut) {
    return crossEntropy(this.forward(srcIds, tgtIn), tgtOut);
  }
}
