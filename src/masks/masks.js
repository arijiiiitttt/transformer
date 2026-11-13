// Masks decide which key positions each query may look at. 1 = allowed, 0 = blocked.
// They are Uint8Array [queries * keys], read by softmax() in autograd/ops.js.
export const PAD = 0;
export const BOS = 1;
export const EOS = 2;

/** Block attention to padding tokens in the keys. Result shape: [nQueries, keyIds.length] */
export function paddingMask(keyIds, nQueries) {
  const nk = keyIds.length;
  const m = new Uint8Array(nQueries * nk);
  for (let i = 0; i < nQueries; i++) for (let j = 0; j < nk; j++) m[i * nk + j] = keyIds[j] !== PAD ? 1 : 0;
  return m;
}

/** "Masked Multi-Head Attention": position i may only see positions <= i. */
export function causalMask(n) {
  const m = new Uint8Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) m[i * n + j] = 1;
  return m;
}

/** keep only where both masks allow it */
export function combineMasks(a, b) {
  const m = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) m[i] = a[i] & b[i];
  return m;
}
