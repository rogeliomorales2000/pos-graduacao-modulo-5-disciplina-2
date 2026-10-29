import { log } from "console";
import { Adam } from "./adam";
import { TinyCNN } from "./tiny-cnn";
import { TinyGPT } from "./tiny-gpt";
import { CharTokenizer } from "./tokenizer";
import type { Window } from "./types.ts";
import { shuffle } from "./utils";
import { sampleFromProbs, softmaxRow } from "./cel";
import { nextTick } from "process";


type EpochStats = {
    epoch: number,
    loss: number
}

export function trainGpt(
    model : TinyGPT,
    windowns : Window[],
    options: {
        epochs:number;
        learningRate?: number;
        seed?: number;
        logEvery?:number;
        onEpoch?:(stats:EpochStats) => void,
        maxWindowsPerEpoch ?: number

    }
):EpochStats []{
    const optimizer = new Adam(options.learningRate ?? 0.001)
    const history : EpochStats[] = [];

    const logEvery = options.logEvery ?? 1;
    for(let epoch = 0 ; epoch < options.epochs; epoch++){
        const order = shuffle(
            windowns,
            (options.seed ?? 0) + epoch
        ).slice(0, options.maxWindowsPerEpoch ?? windowns.length)
        let totalLoss = 0
        for(const w of order ){
            const logist = model.forward(w.input)
            model.backward(logist,w.target)
            optimizer.step(model.parameters());
            totalLoss += model.lastLoss;

        }

        const stats = {
            epoch,
            loss: totalLoss / Math.max(1,order.length)
        }
        history.push(stats)

        options.onEpoch?.(stats);
        if(epoch % logEvery == 0 || epoch ==1 || epoch==options.epochs){
            console.log(`
                Epoch ${epoch}/${options.epochs}  loss=${stats.loss.toFixed(4)}
                
                `)
        }
        if(stats.loss < 0.06 && epoch >= 8){
            console.log(`
                Early stop at epoch ${epoch} (loss < 0.06)    
                
             `)
        }
    }
    

    return history
}


export function evaluateGpt(
    model: TinyGPT,
    windows: Window[]
): number{
    if (windows.length == 0 )return 0;
    let total = 0;

    for (const w of windows){
        const logist = model.forward(w.input)
        total += model.lossFn.forward(logist,w.target)

    }


    return total/windows.length
}


export function generate(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: string,
    options:{
        maxNewTokens ?: number;
        temperature ?: number;
        rng?:() =>number;
        stopChars?: string[];
    }
): string {
    const maxNew = options.maxNewTokens ?? 100
    const temperature = options.temperature ?? 0.0
    const rng = options.rng ?? Math.random
    const stopCharts = new Set(options.stopChars ?? [])

    let ids = tokenizer.encode(prompt)

    if(ids.length == 0){
        const space = tokenizer.stoi.get(`\n`) ?? tokenizer.stoi.get(" ") ?? 0;
        ids = [space];
    }
    for(let i = 0;i < maxNew ; i++){
        const ctx = ids.slice(-model.config.contextSize)
        const logits = model.forward(ctx)
        // Logits têm shape [T, vocab]: a próxima previsão está na ÚLTIMA linha
        const lastRow = logits.shape[0]! - 1
        const row:number[]=[]


        for (let v =0 ; v < model.config.vocabSize;v++){
            row.push(logits.get(lastRow,v))
        }
        const scaled = temperature <= 0 
            ? row
            : row.map((x)=> x / Math.max(temperature,1e-6))
        const probs = softmaxRow(scaled)

        const next = sampleFromProbs(probs,temperature <= 0 ? 0:1 ,rng)
        ids.push(next)

        const ch = tokenizer.decode([next])

        if (stopCharts.has(ch)){
            break;
        }
    }

    return tokenizer.decode(ids)
}