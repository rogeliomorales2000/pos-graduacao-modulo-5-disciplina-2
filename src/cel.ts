import { Tensor } from "./tensor";

const EPS = 1e-12;

/**
 * Cross-entropy média sobre uma sequência de previsões de próximo token.
 *
 * logits:  shape [T, vocabSize]  — scores brutos (não probabilidades)
 * targets: comprimento T         — id do token correto em cada posição
 */
export class CrossEntropyLoss {
  private lastLogits: Tensor | null = null;
  private lastTargets: number[] | null = null;
  private lastProbs: Tensor | null = null;  // softmax cacheado pro backward

  forward(logits: Tensor, targets: number[]): number {
    if (logits.rank !== 2) {
      throw new Error("CrossEntropy expects logits [T, vocab].");
    }
    const [T, V] = logits.shape as [number, number];
    if (targets.length !== T) {
      throw new Error(`Expected ${T} targets, got ${targets.length}.`);
    }

    // Guarda estado para o backward (gradiente usa probs e targets)
    this.lastLogits = logits;
    this.lastTargets = targets;
    this.lastProbs = Tensor.zeros([T, V]);

    let total = 0;
    for (let t = 0; t < T; t++) {
      const target = targets[t]!;
      if (target < 0 || target >= V) {
        throw new Error(`Target id ${target} out of vocab range 0..${V - 1}.`);
      }

      // log-sum-exp estável: max + log(Σ exp(x − max))
      let max = -Infinity;
      for (let v = 0; v < V; v++) {
        max = Math.max(max, logits.get(t, v));
      }
      let sumExp = 0;
      for (let v = 0; v < V; v++) {
        sumExp += Math.exp(logits.get(t, v) - max);
      }
      const logSumExp = max + Math.log(sumExp);

      // Probabilidades = softmax estável (para o backward)
      for (let v = 0; v < V; v++) {
        this.lastProbs.set(Math.exp(logits.get(t, v) - logSumExp), t, v);
      }

      // NLL do token correto: −(logit_target − logSumExp)
      const logProb = logits.get(t, target) - logSumExp;
      total += -logProb;
    }

    return total / T;  // média sobre a sequência
  }

  /**
   * ∂L/∂logits = (softmax(logits) − one_hot(target)) / T
   * Intuição: sobe a prob do token certo, desce a dos errados.
   */
  backward(): Tensor {
    if (
      this.lastLogits === null ||
      this.lastTargets === null ||
      this.lastProbs === null
    ) {
      throw new Error("Must call forward() before backward().");
    }
    const [T, V] = this.lastLogits.shape as [number, number];
    const grad = this.lastProbs.clone();  // começa em softmax
    for (let t = 0; t < T; t++) {
      const target = this.lastTargets[t]!;
      // one-hot: subtrai 1 na posição do token correto
      grad.set(grad.get(t, target) - 1, t, target);
      // Divide por T porque a loss é a média
      for (let v = 0; v < V; v++) {
        grad.set(grad.get(t, v) / T, t, v);
      }
    }
    return grad;
  }
}

/** Softmax de uma única linha de logits (geração / chat). */
export function softmaxRow(logits: number[]): number[] {
  // Mesma estabilidade: subtrai max antes do exp
  let max = -Infinity;
  for (const x of logits) max = Math.max(max, x);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);  // soma = 1
}

/**
 * Amostra um índice de uma distribuição discreta.
 * temperature → 0 é guloso (greedy); maior = mais aleatório.
 * Analogia: temperatura alta = roleta mais "bagunçada".
 */
export function sampleFromProbs(
  probs: number[],
  temperature: number,
  rng: () => number = Math.random,
): number {
  // Temperatura ≤ 0: escolhe sempre o argmax (determinístico)
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i]! > probs[best]!) best = i;
    }
    return best;
  }

  // Re-temperature via logits ≈ log(p) / T — achata ou afia a distribuição
  const logits = probs.map((p) => Math.log(Math.max(p, EPS)) / temperature);
  const softened = softmaxRow(logits);
  // Amostragem por CDF: sorteia u e anda na acumulada até passar
  const u = rng();
  let cum = 0;
  for (let i = 0; i < softened.length; i++) {
    cum += softened[i]!;
    if (u <= cum) return i;
  }
  return softened.length - 1;  // fallback numérico
}