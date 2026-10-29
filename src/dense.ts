import Tensor from "./tensor";

export class Dense {

    private lastInput?: Tensor;
    gradWeights?: Tensor;
    gradBias?: Tensor;
    constructor(
        public weights: Tensor,
        public bias: Tensor
    ) {}

    get weigths(): Tensor {
        return this.weights;
    }

    set weigths(value: Tensor) {
        this.weights = value;
    }

    forward(inputs: Tensor): Tensor {
        this.lastInput = inputs;

        if (inputs.rank === 1) {
            const output = new Float64Array(this.bias.data.length);
            const inSize = inputs.data.length;

            for (let i = 0; i < output.length; i++) {
                let sum = this.bias.data[i];
                for (let j = 0; j < inSize; j++) {
                    const weightIndex = j * output.length + i;
                    sum += inputs.data[j] * this.weights.data[weightIndex];
                }
                output[i] = sum;
            }

            return new Tensor(output, [output.length]);
        }

        if (inputs.rank === 2) {
            const [rows, inFeatures] = inputs.shape as [number, number];
            const outFeatures = this.bias.data.length;
            const output = Tensor.zeros([rows, outFeatures]);

            for (let r = 0; r < rows; r++) {
                for (let o = 0; o < outFeatures; o++) {
                    let sum = this.bias.data[o];
                    for (let i = 0; i < inFeatures; i++) {
                        const wIdx = i * outFeatures + o;
                        sum += inputs.get(r, i) * this.weights.data[wIdx];
                    }
                    output.set(sum, r, o);
                }
            }

            return output;
        }

        throw new Error("Dense.forward expects a 1D or 2D Tensor");
    }

    backward(gradOutput: Tensor): Tensor {
        const input = this.lastInput!;
        const inFeatures = input.shape[input.rank - 1];
        const outFeatures = this.bias.data.length;

        this.gradWeights = Tensor.zeros(this.weights.shape);
        this.gradBias = Tensor.zeros(this.bias.shape);

        if (input.rank === 1) {
            const gradInput = new Float64Array(inFeatures);

            for (let i = 0; i < outFeatures; i++) {
                this.gradBias.data[i] = gradOutput.data[i];
                for (let j = 0; j < inFeatures; j++) {
                    const wIdx = j * outFeatures + i;
                    this.gradWeights.data[wIdx] = input.data[j] * gradOutput.data[i];
                    gradInput[j] += this.weights.data[wIdx] * gradOutput.data[i];
                }
            }

            return new Tensor(gradInput, [inFeatures]);
        }

        if (input.rank === 2) {
            const [rows, cols] = input.shape as [number, number];
            const gradInput = Tensor.zeros([rows, cols]);

            for (let r = 0; r < rows; r++) {
                for (let o = 0; o < outFeatures; o++) {
                    const gradVal = gradOutput.get(r, o);
                    this.gradBias.data[o] += gradVal;

                    for (let c = 0; c < cols; c++) {
                        const wIdx = c * outFeatures + o;
                        this.gradWeights.data[wIdx] += input.get(r, c) * gradVal;
                        gradInput.set(gradInput.get(r, c) + this.weights.data[wIdx] * gradVal, r, c);
                    }
                }
            }

            return gradInput;
        }

        throw new Error("Dense.backward expects a 1D or 2D Tensor");
    }
}
