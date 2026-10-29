import { RELU, Sigmoid } from "./activation";
import { BinaryCrossEntropy } from "./bce";
import { Dense } from "./dense";
import { Example } from "./example";
import { SGD } from "./sgd";
import Tensor from "./tensor";
import { randomMatrix } from "./utils";
import type { Parameter } from "./types.ts";
import { Adam } from "./adam";

// MPL: Multi Layer Perceptron

type TrainableLayer = {
    forward: (input: Tensor) => Tensor;
    backward: (gradOutput: Tensor) => Tensor;
    weigths?: Tensor;
    bias?: Tensor;
};

export class TreinableMLP {
    dense1: Dense;
    dense2: Dense;
    relu: RELU;
    sigmoid: Sigmoid;
    layers: TrainableLayer[];
    loss: BinaryCrossEntropy;
    lossFn: BinaryCrossEntropy;
    optimizer: Adam
    lastloss = 0;

    constructor(
        inputSize: number,
        hiddenSize: number,
        outputSize: number,
        learningRate: number = 0.5,
        seed: number = 42
    ) {
        this.dense1 = new Dense(
            randomMatrix(inputSize, hiddenSize, seed),
            Tensor.zeros([hiddenSize])
        );

        this.dense2 = new Dense(
            randomMatrix(hiddenSize, outputSize, seed + 1),
            Tensor.zeros([outputSize])
        );

        this.relu = new RELU();
        this.sigmoid = new Sigmoid();
        this.loss = new BinaryCrossEntropy();
        this.lossFn = new BinaryCrossEntropy();
        this.optimizer = new Adam(0.05);
        this.layers = [this.dense1, this.relu, this.dense2, this.sigmoid];
    }

    private toTensor(value: Tensor | number[]): Tensor {
        if (value instanceof Tensor) {
            return value;
        }
        return new Tensor(new Float64Array(value), [value.length]);
    }

    trainStep(example: Example): number {
        const input = this.toTensor(example.inputs as Tensor | number[]);
        const targetValue = typeof example.target === "number" ? example.target : example.target.data[0];

        const prediction = this.forward(input);
        this.lastloss = this.loss.forward(prediction.data[0], targetValue);

        const safePred = Math.max(1e-8, Math.min(1 - 1e-8, prediction.data[0]));
        const gradOutput = new Tensor(new Float64Array([(safePred - targetValue) / (safePred * (1 - safePred))]), [1]);
        let grad = gradOutput;

        for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad);
        }

        this.optimizer.step(this.parameters());
        return this.lastloss;
    }

    forward(input: Tensor): Tensor {
        let x = input;
        for (const layer of this.layers) {
            x = layer.forward(x);
        }
        return x;
    }

    backward(prediction: Tensor, target: number | Tensor) {
        const targetValue = typeof target === "number" ? target : target.data[0];
        this.lastloss = this.lossFn.forward(prediction.data[0], targetValue);

        const grad = new Tensor(new Float64Array([prediction.data[0] - targetValue]), [1]);
        let currentGrad = grad;

        for (let i = this.layers.length - 1; i >= 0; i--) {
            currentGrad = this.layers[i].backward(currentGrad);
        }

        return currentGrad;
    }

    parameters(): Parameter[] {
        return [
            { name: "dense1.weights", data: this.dense1.weigths, grad: this.dense1.gradWeights ?? Tensor.zeros(this.dense1.weigths.shape) },
            { name: "dense1.bias", data: this.dense1.bias, grad: this.dense1.gradBias ?? Tensor.zeros(this.dense1.bias.shape) },
            { name: "dense2.weights", data: this.dense2.weigths, grad: this.dense2.gradWeights ?? Tensor.zeros(this.dense2.weigths.shape) },
            { name: "dense2.bias", data: this.dense2.bias, grad: this.dense2.gradBias ?? Tensor.zeros(this.dense2.bias.shape) }
        ];
    }
}