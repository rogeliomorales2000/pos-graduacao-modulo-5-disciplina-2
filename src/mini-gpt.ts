import fs from "node:fs";
import path from "node:path";

import {
  branchingPrefix,
  cueFromQuestion,
  DocStore,
  factsStartingWith,
  fitToVocab,
  RetrievedChunk,
  retrieveWithModel,
} from "./rag";
import { sentenceWindows as setenceWindows } from "./sentence-windowns";
import { TextDataset } from "./text-dataset";
import { defaultGptConfig, TinyGPT, TinyGptConfig } from "./tiny-gpt";
import { CharTokenizer } from "./tokenizer";
import { evaluateGpt, generate, trainGpt } from "./train-gpt";

// Formato do checkpoint JSON: config + vocab + pesos nomeados
export interface TrainedGptArtifact {
  config: TinyGptConfig;
  tokenizer: { vocab: string[] };
  parameters: Record<string, number[]>;
}

/** Lê e concatena todos os .txt de um split (train/validation/test). */
export function loadSplitText(datasetRoot: string, split: string): string {
  const dir = path.join(datasetRoot, split);
  if (!fs.existsSync(dir)) return ""; // split ausente = string vazia
  const files = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".txt"))
    .sort(); // ordem estável entre máquinas
  return files
    .map((f) => fs.readFileSync(path.join(dir, f), "utf8"))
    .join("\n");
}

/** Pipeline completo: carrega dados → treina → (opcional) salva checkpoint. */
export function trainMiniGpt(options: {
  datasetRoot: string;
  epochs?: number;
  learningRate?: number;
  contextSize?: number;
  dim?: number;
  modelPath?: string;
  maxWindowsPerEpoch?: number;
}): TinyGPT {
  const contextSize = options.contextSize ?? 96;
  const trainText = loadSplitText(options.datasetRoot, "train");
  // Sem corpus = aluno esqueceu de gerar o dataset
  if (trainText.trim().length === 0) {
    throw new Error(
      `No .txt files found under ${options.datasetRoot}/train. Run: npm run generate-dataset`,
    );
  }
  // Vocab unificado a partir de train+val+test
  const ds = TextDataset.fromSplits(
    {
      train: trainText,
      validation: loadSplitText(options.datasetRoot, "validation"),
      test: loadSplitText(options.datasetRoot, "test"),
    },
    contextSize,
  );

  console.log(ds.describe());
  console.log("");

  // Modelo um pouco maior que o default didático (dim 64)
  const model = new TinyGPT({
    ...defaultGptConfig(ds.tokenizer.vocabSize),
    contextSize,
    dim: options.dim ?? 64,
    hiddenMul: 2,
    seed: 42,
  });

  console.log(model.summary());
  console.log("\nTraining next-token on documents (no question/answer pairs)…\n");

  // Prefere documents.txt; senão usa todo o train
  const docsPath = path.join(options.datasetRoot, "train", "documents.txt");
  const docsText = fs.existsSync(docsPath)
    ? fs.readFileSync(docsPath, "utf8")
    : trainText;
  // Janelas por frase — alinhadas com generate() seeded no início do fato
  const factWindows = setenceWindows(docsText, ds.tokenizer, contextSize);
  if (factWindows.length === 0) {
    throw new Error(
      "No sentences found in training documents. Run: npm run generate-dataset",
    );
  }
  // Repete 20× para ter volume suficiente de exemplos por época
  const trainWindows = Array.from({ length: 20 }, () => factWindows).flat();
  console.log(
    `Train examples: ${trainWindows.length} (from ${factWindows.length} document sentences)\n`,
  );

  trainGpt(model, trainWindows, {
    epochs: options.epochs ?? 60,
    learningRate: options.learningRate ?? 0.004,
    seed: 42,
    maxWindowsPerEpoch : options.maxWindowsPerEpoch ?? trainWindows.length,
  });

  // Avaliação em val/test (só reporta; não treina)
  const valWindows = setenceWindows(
    loadSplitText(options.datasetRoot, "validation"),
    ds.tokenizer,
    contextSize,
  );
  if (valWindows.length > 0) {
    const valLoss = evaluateGpt(model, valWindows);
    console.log(`\nValidation loss: ${valLoss.toFixed(4)}`);
  }

  const testWindows = setenceWindows(
    loadSplitText(options.datasetRoot, "test"),
    ds.tokenizer,
    contextSize,
  );
  if (testWindows.length > 0) {
    const testLoss = evaluateGpt(model, testWindows);
    console.log(`Test loss:       ${testLoss.toFixed(4)}`);
  }

  // Persiste modelo + tokenizer juntos (vocab dessincronizado = desastre)
  if (options.modelPath) {
    saveModel(model, ds.tokenizer, options.modelPath);
    console.log(`\nSaved model → ${options.modelPath}`);
  }

  return model;
}

/** Serializa config + vocab + todos os pesos num JSON legível. */
export function saveModel(
  model: TinyGPT,
  tokenizer: CharTokenizer,
  filePath: string,
): void {
  const artifact: TrainedGptArtifact = {
    config: { ...model.config },
    tokenizer: tokenizer.toJSON(),
    parameters: {},
  };
  // Copia cada parâmetro nomeado (tokenEmb, posEmb, attn.*, ffn.*, lmHead.*)
  for (const p of model.parameters()) {
    const values =
      typeof p.data === "number"
        ? [p.data]
        : Array.from(p.data.data);
    artifact.parameters[p.name] = values;
  }
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(artifact, null, 2));
}

/** Recria TinyGPT + CharTokenizer a partir do checkpoint. */
export function loadModel(filePath: string): {
  model: TinyGPT;
  tokenizer: CharTokenizer;
} {
  const artifact = JSON.parse(
    fs.readFileSync(filePath, "utf8"),
  ) as TrainedGptArtifact;
  const tokenizer = CharTokenizer.fromJSON(artifact.tokenizer);
  const model = new TinyGPT(artifact.config);
  // Restaura pesos pelo nome — se faltar algum, falha cedo
  for (const p of model.parameters()) {
    const values = artifact.parameters[p.name];
    if (!values) {
      throw new Error(`Missing parameter ${p.name} in ${filePath}`);
    }
    if (typeof p.data === "number") {
      p.data = values[0] ?? 0;
      continue;
    }
    for (let i = 0; i < values.length; i++) {
      p.data.data[i] = values[i]!;
    }
  }
  return { model, tokenizer };
}

/**
 * Mapeia texto livre para o vocab de caracteres (lowercase, descarta OOV).
 */
export function fitPromptToVocab(
  tokenizer: CharTokenizer,
  text: string,
): string {
  return fitToVocab(tokenizer, text); // delega ao helper do cap. 18
}

/**
 * Uma rodada de chat: opcionalmente recupera conhecimento (RAG) e gera reply.
 * Vive em mini-gpt.ts (NÃO em rag.ts) — rag só fornece as peças.
 */
export function chatOnce(
  model: TinyGPT,
  tokenizer: CharTokenizer,
  history: string,
  userLine: string,
  options: {
    maxNewTokens?: number;
    temperature?: number;
    /** Store opcional — cue matching + fontes relacionadas. */
    store?: DocStore;
    topK?: number;
    rng?: () => number;
  } = {},
): { reply: string; history: string; retrieved: RetrievedChunk[]; cue: string } {
  // Filtra caracteres fora do vocab (evita crash no encode)
  const fitted = fitPromptToVocab(tokenizer, userLine);
  if (!fitted) {
    throw new Error(
      "Message has no characters in the model vocab (try lowercase letters/spaces).",
    );
  }

  const temperature = options.temperature ?? 0.8;
  const rng = options.rng ?? Math.random;

  let retrieved: RetrievedChunk[] = [];
  let prompt: string;
  if (options.store) {
    // Com RAG: busca chunks relevantes e monta um cue de geração
    retrieved = retrieveWithModel(
      model,
      tokenizer,
      options.store,
      fitted,
      options.topK ?? 3,
    );
    const knowledge = options.store.chunks.map((c) => c.text).join(" ");
    prompt = cueFromQuestion(fitted, knowledge);
    // Se vários fatos começam igual, escolhe um e usa prefixo único
    const facts = factsStartingWith(prompt, knowledge);
    if (facts.length > 1) {
      const fact = facts[Math.floor(rng() * facts.length)]!;
      prompt = branchingPrefix(fact, facts);
    }
  } else {
    // Sem store: gera direto a partir do texto do usuário
    prompt = fitted;
  }
  if (!prompt) prompt = fitted;

  // Gera até \n (uma "linha" de resposta)
  const full = generate(model, tokenizer, prompt, {
    maxNewTokens: options.maxNewTokens ?? 80,
    temperature,
    rng,
    stopChars: ["\n"],
  });

  // Limpa: descarta o cue/prompt inicial, mantém a primeira linha e
  // corta no primeiro ponto (resposta curta)
  const generated = full.startsWith(prompt) ? full.slice(prompt.length) : full;
  let reply = generated.split("\n")[0]!.trim();
  const dot = reply.indexOf(".");
  if (dot >= 0) reply = reply.slice(0, dot + 1).trim();

  // Atualiza histórico no formato You:/Bot:
  const nextHistory = `${history}You: ${fitted}\nBot: ${reply}\n`;
  return { reply, history: nextHistory, retrieved, cue: prompt };
}

/** Dump didático: summary + ids + shapes do forward (npm run inspect). */
export function formatInspect(
  model: TinyGPT,
  tokenizer: CharTokenizer,
  prompt: string,
): string {
  const ids = tokenizer.encode(prompt).slice(-model.config.contextSize);
  const logits = model.forward(ids, true); // recordTrace=true
  const traceLines = model.trace.map(
    (t) => `  ${t.name}: [${(t.shape ?? []).join(", ")}]`,
  );

  const lines = [
    model.summary(),
    "",
    `Prompt: ${JSON.stringify(prompt)}`,
    `Token ids (${ids.length}): [${ids.slice(0, 16).join(", ")}${ids.length > 16 ? ", ..." : ""}]`,
    "",
    "Forward shapes:",
    ...traceLines,
    "",
    `Logits shape: [${(logits.shape ?? []).join(", ")}]`,
  ];
  return lines.join("\n");
}