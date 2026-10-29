import test from 'node:test';
import assert from 'node:assert/strict';
import { CharTokenizer } from './tokenizer';

test('CharTokenizer encodes and decodes a vocab', () => {
  const tokenizer = new CharTokenizer(['r', 'o', 'g', 'e']);

  assert.deepEqual(tokenizer.encode('roge'), [0, 1, 2, 3]);
  assert.equal(tokenizer.decode([0, 1, 2, 3]), 'roge');
});
