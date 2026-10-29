/**
 * Chapter 18 — Retrieve Then Generate (tiny RAG)
 *
 * Tokenize → embed chunks → cosine similarity search → stuff top
 * chunks into the prompt → generate the answer (next tokens).
 */

import fs from "node:fs";
import path from "node:path";
import { Embedding } from "./embeddings";
import { TinyGPT } from "./tiny-gpt";
import { CharTokenizer } from "./tokenizer";

export interface DocChunk {
  id: string;
  source: string;
  text: string;
}

export interface DocStore {
  dim: number;
  chunks: DocChunk[];
  /** One vector per chunk, length `dim`. */
  vectors: number[][];
}

export interface RetrievedChunk {
  chunk: DocChunk;
  score: number;
}

/** Split text into sentence-sized knowledge chunks. */
export function chunkText(
  text: string,
  options: { size?: number; overlap?: number; source?: string } = {},
): DocChunk[] {
  const source = options.source ?? "doc";
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return [];

  const sentences = normalized
    .split(/(?<=\.)\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 8);

  if (sentences.length === 0) {
    return [{ id: `${source}#0`, source, text: normalized.slice(0, 90) }];
  }

  return sentences.map((sentence, i) => ({
    id: `${source}#${i}`,
    source,
    text: sentence,
  }));
}

/** Mean-pool token embedding rows → one vector for the string. */
export function embedText(
  embedding: Embedding,
  tokenizer: CharTokenizer,
  text: string,
): number[] {
  const fitted = fitToVocab(tokenizer, text);
  if (!fitted) {
    const dim = embedding.weight.shape[1]!;
    return Array(dim).fill(0);
  }
  const ids = tokenizer.encode(fitted);
  const mat = embedding.forward(ids);
  const T = mat.shape[0]!;
  const dim = mat.shape[1]!;
  const out = Array(dim).fill(0) as number[];
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < dim; d++) {
      out[d]! += mat.get(t, d);
    }
  }
  for (let d = 0; d < dim; d++) out[d]! /= T;
  return out;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom === 0) return 0;
  return dot / denom;
}

/** Fraction of content words in the query that appear in the document. */
export function lexicalOverlap(query: string, doc: string): number {
  const stop = new Set([
    "how",
    "many",
    "does",
    "what",
    "when",
    "which",
    "the",
    "a",
    "an",
    "is",
    "are",
    "do",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "and",
  ]);
  const qWords = [
    ...new Set(
      query
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter((w) => w.length > 1 && !stop.has(w)),
    ),
  ];
  if (qWords.length === 0) return 0;
  const docLower = doc.toLowerCase();
  let found = 0;
  for (const w of qWords) {
    if (docLower.includes(w)) found += 1;
  }
  // Prefer denser absolute overlap so short titles don't beat fact sentences.
  return found / qWords.length + 0.05 * found;
}

export function fitToVocab(tokenizer: CharTokenizer, text: string): string {
  let out = "";
  for (const ch of text.toLowerCase()) {
    if (tokenizer.stoi.has(ch)) out += ch;
  }
  return out.replace(/\s+/g, " ").trim();
}

const CUE_STOP = new Set([
  "how",
  "many",
  "does",
  "what",
  "when",
  "which",
  "the",
  "a",
  "an",
  "is",
  "are",
  "do",
  "of",
  "to",
  "for",
  "in",
  "on",
  "at",
  "and",
  "explain",
  "mean",
  "means",
]);

/**
 * Pull a generation cue from the question — a short prefix the LM was
 * trained to continue. Not a retrieved sentence pasted as the reply.
 */
export function cueFromQuestion(question: string, knowledge: string): string {
  const q = question
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const sentences = knowledge
    .toLowerCase()
    .split(/(?<=\.)\s+/)
    .map((s) => s.trim().replace(/\.$/, ""))
    .filter(Boolean);
  if (!q) return q;
  const qWords = q.split(/\s+/).filter(Boolean);
  const content = qWords.filter((w) => w.length > 1 && !CUE_STOP.has(w));

  const starts = (phrase: string): boolean =>
    sentences.some((s) => s === phrase || s.startsWith(`${phrase} `));

  // Longest question n-gram that begins a stored fact (the/a allowed).
  for (let n = Math.min(3, qWords.length); n >= 1; n--) {
    for (let i = 0; i <= qWords.length - n; i++) {
      const phrase = qWords.slice(i, i + n).join(" ");
      const hasContent = phrase
        .split(/\s+/)
        .some((w) => w.length > 1 && !CUE_STOP.has(w));
      if (!hasContent) continue;
      if (starts(phrase)) return phrase;
    }
  }
  for (const w of content) {
    if (starts(`the ${w}`)) return `the ${w}`;
    if (starts(`a ${w}`)) return `a ${w}`;
  }
  // Clue is mid-sentence: recall the fact, start generating from its opening.
  for (const w of content) {
    const sent = sentences.find((s) => hasPhrase(s, w));
    if (!sent) continue;
    const sw = sent.split(/\s+/);
    if (sw.length >= 2 && CUE_STOP.has(sw[0]!)) return `${sw[0]} ${sw[1]}`;
    if (sw[0]) return sw[0];
  }
  return content[0] ?? q;
}

/** Facts the model learned that begin with this cue. */
export function factsStartingWith(cue: string, knowledge: string): string[] {
  const c = cue.toLowerCase().trim();
  if (!c) return [];
  return knowledge
    .toLowerCase()
    .split(/(?<=\.)\s+/)
    .map((s) => s.trim())
    .filter((s) => s === c || s.startsWith(`${c} `) || s.startsWith(`${c}.`));
}

/**
 * Shortest prefix that uniquely identifies one fact among siblings.
 * Sampling this prefix is how a tiny LM chooses among learned wordings.
 */
export function branchingPrefix(fact: string, siblings: string[]): string {
  const words = fact.replace(/\.$/, "").split(/\s+/).filter(Boolean);
  const others = siblings.filter((s) => s !== fact);
  for (let n = 2; n <= words.length; n++) {
    const prefix = words.slice(0, n).join(" ");
    const clash = others.some(
      (o) =>
        o === prefix || o.startsWith(`${prefix} `) || o.startsWith(`${prefix}.`),
    );
    if (!clash) return prefix;
  }
  return words.slice(0, Math.min(3, words.length)).join(" ");
}

function hasPhrase(doc: string, phrase: string): boolean {
  const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`).test(doc);
}

/** Embed every chunk with the model's token embedding table. */
export function buildDocStore(
  chunks: DocChunk[],
  embedding: Embedding,
  tokenizer: CharTokenizer,
): DocStore {
  const dim = embedding.weight.shape[1]!;
  const vectors = chunks.map((c) => embedText(embedding, tokenizer, c.text));
  return { dim, chunks, vectors };
}

export function retrieve(
  query: string,
  store: DocStore,
  embedding: Embedding,
  tokenizer: CharTokenizer,
  topK = 2,
): RetrievedChunk[] {
  const q = embedText(embedding, tokenizer, query);
  const scored = store.chunks.map((chunk, i) => {
    const cos = cosineSimilarity(q, store.vectors[i]!);
    const lex = lexicalOverlap(query, chunk.text);
    // Lexical overlap dominates on a tiny char model; cosine is secondary.
    return { chunk, score: 0.35 * cos + 0.65 * lex };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(1, topK));
}

/**
 * Sample one retrieved passage using score softmax.
 * temperature → 0 always takes the top hit; higher mixes near-ties.
 */
export function sampleRetrieved(
  hits: RetrievedChunk[],
  temperature: number,
  rng: () => number = Math.random,
): RetrievedChunk {
  if (hits.length === 0) {
    throw new Error("sampleRetrieved needs at least one hit");
  }
  if (hits.length === 1 || temperature <= 0) return hits[0]!;
  const scores = hits.map((h) => h.score);
  const max = Math.max(...scores);
  const weights = scores.map((s) => Math.exp((s - max) / temperature));
  const sum = weights.reduce((a, b) => a + b, 0);
  let u = rng() * sum;
  for (let i = 0; i < hits.length; i++) {
    u -= weights[i]!;
    if (u <= 0) return hits[i]!;
  }
  return hits[hits.length - 1]!;
}

/** First few words of a passage — seed so the LM continues from learned text. */
export function passageSeed(text: string, words = 2): string {
  const parts = text.trim().split(/\s+/).filter(Boolean);
  if (parts.length <= 1) return text.trim();
  const keep = Math.min(words, parts.length - 1);
  return parts.slice(0, keep).join(" ");
}

export function loadKnowledgeDir(dir: string): DocChunk[] {
  if (!fs.existsSync(dir)) return [];
  const files = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".txt"))
    .sort();
  const chunks: DocChunk[] = [];
  for (const f of files) {
    const text = fs.readFileSync(path.join(dir, f), "utf8");
    chunks.push(...chunkText(text, { source: f.replace(/\.txt$/, "") }));
  }
  return chunks;
}

export function saveDocStore(store: DocStore, filePath: string): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(store, null, 2));
}

export function loadDocStore(filePath: string): DocStore {
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as DocStore;
}

/** Convenience: retrieve with the live TinyGPT token embedding. */
export function retrieveWithModel(
  model: TinyGPT,
  tokenizer: CharTokenizer,
  store: DocStore,
  query: string,
  topK = 2,
): RetrievedChunk[] {
  return retrieve(query, store, model.tokenEmb, tokenizer, topK);
}