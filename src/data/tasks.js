// Tiny sequence-to-sequence tasks. A Transformer that learns them has learned to
// route information with attention, which is exactly what the architecture is for.
//   reverse : "12345" -> "54321"   needs positional attention, left/right flip
//   copy    : "12345" -> "12345"   the easiest warm-up
//   sort    : "31415" -> "11345"   needs the encoder to compare every pair of tokens
import { mulberry32, randint } from '../utils/rng.js';

const transforms = {
  reverse: (s) => [...s].reverse().join(''),
  copy: (s) => s,
  sort: (s) => [...s].sort().join(''),
};

export const TASK_NAMES = Object.keys(transforms);

export function makeSample(task, alphabet, minLen, maxLen, rng) {
  const f = transforms[task];
  if (!f) throw new Error(`unknown task "${task}". Choose from: ${TASK_NAMES.join(', ')}`);
  const n = randint(rng, minLen, maxLen + 1);
  let src = '';
  for (let i = 0; i < n; i++) src += alphabet[randint(rng, 0, alphabet.length)];
  return { src, tgt: f(src) };
}

export function makeDataset(task, alphabet, minLen, maxLen, count, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: count }, () => makeSample(task, alphabet, minLen, maxLen, rng));
}
