/**
 * CLI de chat do mini-GPT (RAG opcional).
 *
 * Uso pontual:
 *   npm run chat -- --prompt "how many remote days does acme allow?"
 *
 * Modo interativo:
 *   npm run chat
 *
 * Opções:
 *   --prompt <texto>       Responde uma vez e encerra.
 *   --model <caminho>      Checkpoint do modelo (padrão: models/mini-gpt.json).
 *   --store <caminho>      Índice RAG (padrão: models/doc-store.json).
 *   --no-rag               Não usa o índice (gera direto do prompt).
 *   --max-new-tokens <n>   Máximo de tokens gerados (padrão: 80).
 *   --temperature <n>      Criatividade da amostragem (padrão: 0.8).
 *   --top-k <n>            Chunks recuperados pelo RAG (padrão: 3).
 *   --show-sources         Mostra as fontes recuperadas.
 *   --help                 Mostra esta ajuda.
 */

import fs from "node:fs";
import path from "node:path";
import * as readline from "node:readline/promises";
import process from "node:process";

import { chatOnce, loadModel } from "../../src/mini-gpt";
import { DocStore, loadDocStore } from "../../src/rag";
import { CharTokenizer } from "../../src/tokenizer";
import { TinyGPT } from "../../src/tiny-gpt";

interface ChatOptions {
  prompt?: string;
  modelPath: string;
  storePath: string;
  useRag: boolean;
  maxNewTokens: number;
  temperature: number;
  topK: number;
  showSources: boolean;
}

const HELP = `
Chat do mini-GPT

Uso:
  npm run chat                          # modo interativo
  npm run chat -- --prompt "pergunta"   # responde uma vez e encerra

Opções:
  --prompt <texto>       Responde uma vez e encerra.
  --model <caminho>      Checkpoint do modelo (padrão: models/mini-gpt.json).
  --store <caminho>      Índice RAG (padrão: models/doc-store.json).
  --no-rag               Não usa o índice (gera direto do prompt).
  --max-new-tokens <n>   Máximo de tokens gerados (padrão: 80).
  --temperature <n>      Criatividade da amostragem (padrão: 0.8).
  --top-k <n>            Chunks recuperados pelo RAG (padrão: 3).
  --show-sources         Mostra as fontes recuperadas.
  --help                 Mostra esta ajuda.
`.trim();

function readValue(argv: string[], index: number, flag: string): string {
  const value = argv[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`Faltou o valor de ${flag}.`);
  }
  return value;
}

function parseArgs(argv: string[]): ChatOptions {
  const root = process.cwd();
  const options: ChatOptions = {
    modelPath: path.join(root, "models", "mini-gpt.json"),
    storePath: path.join(root, "models", "doc-store.json"),
    useRag: true,
    maxNewTokens: 80,
    temperature: 0.8,
    topK: 3,
    showSources: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    const [flag, inlineValue] = arg.includes("=")
      ? [arg.slice(0, arg.indexOf("=")), arg.slice(arg.indexOf("=") + 1)]
      : [arg, undefined];

    const value = (): string => {
      if (inlineValue !== undefined) return inlineValue;
      const v = readValue(argv, i, flag);
      i += 1;
      return v;
    };

    switch (flag) {
      case "--prompt":
      case "-p":
        options.prompt = value();
        break;
      case "--model":
        options.modelPath = path.resolve(root, value());
        break;
      case "--store":
        options.storePath = path.resolve(root, value());
        break;
      case "--no-rag":
        options.useRag = false;
        break;
      case "--max-new-tokens":
        options.maxNewTokens = Number(value());
        break;
      case "--temperature":
        options.temperature = Number(value());
        break;
      case "--top-k":
        options.topK = Number(value());
        break;
      case "--show-sources":
        options.showSources = true;
        break;
      case "--help":
      case "-h":
        console.log(HELP);
        process.exit(0);
      default:
        throw new Error(`Opção desconhecida: ${arg}\n\n${HELP}`);
    }
  }

  if (
    !Number.isFinite(options.maxNewTokens) ||
    options.maxNewTokens <= 0 ||
    !Number.isInteger(options.maxNewTokens)
  ) {
    throw new Error("--max-new-tokens deve ser um inteiro maior que zero.");
  }
  if (!Number.isFinite(options.temperature) || options.temperature < 0) {
    throw new Error("--temperature deve ser um número maior ou igual a zero.");
  }
  if (!Number.isFinite(options.topK) || options.topK <= 0) {
    throw new Error("--top-k deve ser um número maior que zero.");
  }
  if (options.prompt !== undefined && options.prompt.trim() === "") {
    throw new Error("--prompt não pode ser vazio.");
  }

  return options;
}

function loadStore(
  model: TinyGPT,
  tokenizer: CharTokenizer,
  options: ChatOptions,
): DocStore | undefined {
  if (!options.useRag) return undefined;

  if (!fs.existsSync(options.storePath)) {
    console.warn(
      `Aviso: índice RAG não encontrado em ${options.storePath}; ` +
        "seguindo sem RAG. Rode `npm run train:mini-gpt` para criá-lo.",
    );
    return undefined;
  }

  const store = loadDocStore(options.storePath);
  if (store.dim !== model.tokenEmb.weight.shape[1]) {
    console.warn(
      `Aviso: índice RAG tem dim ${store.dim}, mas o modelo tem dim ` +
        `${model.tokenEmb.weight.shape[1]}; seguindo sem RAG.`,
    );
    return undefined;
  }
  return store;
}

function printSources(
  retrieved: ReturnType<typeof chatOnce>["retrieved"],
): void {
  if (retrieved.length === 0) return;
  console.log("Fontes:");
  for (const hit of retrieved) {
    console.log(
      `  - ${hit.chunk.source} (score ${hit.score.toFixed(3)}): ${hit.chunk.text}`,
    );
  }
}

async function runOnce(
  model: TinyGPT,
  tokenizer: CharTokenizer,
  store: DocStore | undefined,
  options: ChatOptions,
): Promise<void> {
  const result = chatOnce(model, tokenizer, "", options.prompt!, {
    store,
    maxNewTokens: options.maxNewTokens,
    temperature: options.temperature,
    topK: options.topK,
  });

  console.log(`Bot: ${result.reply}`);
  if (options.showSources) printSources(result.retrieved);
}

async function runInteractive(
  model: TinyGPT,
  tokenizer: CharTokenizer,
  store: DocStore | undefined,
  options: ChatOptions,
): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  let history = "";
  console.log("Chat do mini-GPT. Digite /exit (ou Ctrl+C) para sair.\n");

  try {
    while (true) {
      let line: string;
      try {
        line = (await rl.question("You: ")).trim();
      } catch {
        break; // Ctrl+C / EOF
      }
      if (!line) continue;
      if (line === "/exit" || line === "/quit") break;

      try {
        const result = chatOnce(model, tokenizer, history, line, {
          store,
          maxNewTokens: options.maxNewTokens,
          temperature: options.temperature,
          topK: options.topK,
        });
        history = result.history;
        console.log(`Bot: ${result.reply}`);
        if (options.showSources) printSources(result.retrieved);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`Erro: ${message}`);
      }
    }
  } finally {
    rl.close();
  }
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));

  if (!fs.existsSync(options.modelPath)) {
    throw new Error(
      `Modelo não encontrado em ${options.modelPath}. ` +
        "Rode `npm run train:mini-gpt` primeiro.",
    );
  }

  const { model, tokenizer } = loadModel(options.modelPath);
  const store = loadStore(model, tokenizer, options);

  if (options.prompt !== undefined) {
    await runOnce(model, tokenizer, store, options);
    return;
  }
  await runInteractive(model, tokenizer, store, options);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Erro: ${message}`);
  process.exitCode = 1;
});
