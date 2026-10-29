import type { Neuron } from "./neuron";
import type { Example } from "./example";
import Tensor from "./tensor";

export function sigmoid(x: number): number {
    // Sigmoid activation function que tranforma qualquer numero em um valor entre 0 e 1 
    return 1 / (1 + Math.exp(-x));
}

export function acuracy (
    neuron: Neuron,
    examples: Example[],
    threshold: number = 0.5 // Converter a previsao do neurorio em uma classificacao binaria 0 ou 1 binaria
){
    if (examples.length === 0) {
        return 0;
    }

    let correct = 0;

    for (const example of examples) {
        const prediction = neuron.forward(example.inputs);
        const guess = prediction >= threshold ? 1 : 0;
        if (guess === example.target) {
            correct++;
        }
    }

    return correct / examples.length;
}


export type Parameter = {
    name: string,
    data: number | Tensor,
    grad: number | Tensor
};

/** Janela de treino next-token: entrada e alvo deslocado em 1. */
export type Window = {
    input: number[],
    target: number[]
};


export type Nullable < T > = T | null

