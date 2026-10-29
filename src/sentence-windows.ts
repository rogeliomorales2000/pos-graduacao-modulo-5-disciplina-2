import { CharTokenizer } from "./tokenizer";
import type { Window } from "./types.ts";

export function sentenceWindows(
  text: string,
  tokenizer: CharTokenizer,
  contextSize: number,
): Window[] {
  const sentences = text
    .split(/(?<=\.)\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const windows: Window[] = [];

  for (const s of sentences) {
    const normalized = s.endsWith("\n") ? s : `${s}\n`;
    const ids = tokenizer.encode(normalized);

    if (ids.length < 2) {
      continue;
    }

    if (ids.length - 1 <= contextSize) {
      windows.push({
        input: ids.slice(0, -1),
        target: ids.slice(1),
      });
    } else {
      const slice = ids.slice(0, contextSize + 1);
      windows.push({
        input: slice.slice(0, -1),
        target: slice.slice(1),
      });
    }
  }

  return windows;
}

export const setenceWindows = sentenceWindows;
