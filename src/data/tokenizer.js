// Character-level tokenizer. Ids 0,1,2 are the special tokens PAD, BOS, EOS.
// Every character in `alphabet` gets the next id (3, 4, ...).
import { BOS, EOS, PAD } from '../masks/masks.js';

export class CharTokenizer {
  constructor(alphabet) {
    this.alphabet = alphabet;
    this.index = new Map([...alphabet].map((c, i) => [c, i + 3]));
    this.vocabSize = alphabet.length + 3;
  }

  encode(text) {
    return [...text].map((c) => {
      const id = this.index.get(c);
      if (id === undefined) throw new Error(`character "${c}" is not in the alphabet "${this.alphabet}"`);
      return id;
    });
  }

  /** ids -> text. Stops at EOS and skips PAD/BOS. */
  decode(ids) {
    let s = '';
    for (const id of ids) {
      if (id === EOS) break;
      if (id === PAD || id === BOS) continue;
      s += this.alphabet[id - 3];
    }
    return s;
  }
}
