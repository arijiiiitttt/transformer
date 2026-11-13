// Generation (inference). The encoder runs once; then the decoder produces one
// token at a time, feeding its own previous outputs back in until it emits EOS.
import { noGrad } from '../autograd/tensor.js';
import { BOS, EOS } from '../masks/masks.js';

/**
 * @param model  Transformer
 * @param srcIds token ids of the input
 * @param opts   { maxNew, temperature (0 = greedy), topK, rng }
 * @returns { ids, steps: [{ id, prob }] }
 */
export function generateIds(model, srcIds, { maxNew = 24, temperature = 0, topK = 0, rng = Math.random } = {}) {
  return noGrad(() => {
    const memory = model.encode(srcIds);
    const ids = [BOS];
    const steps = [];
    const limit = Math.min(maxNew, model.cfg.maxLen - 1);
    for (let t = 0; t < limit; t++) {
      const hidden = model.decode(ids, memory, srcIds);
      const V = model.cfg.vocabSize;
      const probs = model.head.probabilities(hidden).data.slice((ids.length - 1) * V, ids.length * V);
      const id = temperature > 0 ? sample(probs, temperature, topK, rng) : argmax(probs);
      steps.push({ id, prob: probs[id] });
      if (id === EOS) break;
      ids.push(id);
    }
    return { ids: ids.slice(1), steps };
  });
}

/** text in, text out */
export function generateText(model, tokenizer, text, opts = {}) {
  const { ids } = generateIds(model, tokenizer.encode(text), opts);
  return tokenizer.decode(ids);
}

function argmax(p) {
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  return best;
}

function sample(probs, temperature, topK, rng) {
  // re-weight with temperature, optionally keep only the top-k tokens
  let items = Array.from(probs, (p, i) => [i, Math.pow(Math.max(p, 1e-12), 1 / temperature)]);
  if (topK > 0) items = items.sort((a, b) => b[1] - a[1]).slice(0, topK);
  const total = items.reduce((s, [, w]) => s + w, 0);
  let r = rng() * total;
  for (const [i, w] of items) {
    r -= w;
    if (r <= 0) return i;
  }
  return items[items.length - 1][0];
}
