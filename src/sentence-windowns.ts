import { CharTokenizer } from "./tokenizer";
import type { Window } from "./types";

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

    // precisamos de pelo menos input + 1 target
    if (ids.length < 2) {
      continue;
    }

    if (ids.length - 1 <= contextSize) {
      // Frase cabe no contexto: use a frase inteira deslocada
      windows.push({
        input: ids.slice(0, -1),
        target: ids.slice(1)
      });
    } else {
      // Frase maior que o context: corta os primeiros contextSize+1 tokens
      const slice = ids.slice(0, contextSize + 1);
      windows.push({
        input: slice.slice(0, -1),
        target: slice.slice(1)
      });
    }
  }

  return windows;
}