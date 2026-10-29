import Tensor from "./tensor";
import type { Parameter } from "./types.ts";

export interface Conv2DOptions {
    stride?: number;
    padding?: number;
    seed?: number;
}

function randomKernel(size: number, seed: number): Tensor {
    let state = seed;
    const data = new Float64Array(size * size);

    for (let i = 0; i < data.length; i++) {
        state = (state * 1664525 + 1013904223) % 4294967296;
        const u = state / 4294967296;
        data[i] = (u - 0.5) * 0.1;
    }

    return new Tensor(new Float64Array(data), [size, size]);
}

function convolve2d(image: Tensor, kernel: Tensor, options: { stride?: number; padding?: number } = {}): Tensor {
    const stride = options.stride ?? 1;
    const padding = options.padding ?? 0;
    const [h, w] = image.shape as [number, number];
    const [kh, kw] = kernel.shape as [number, number];

    const padded = Tensor.zeros([h + 2 * padding, w + 2 * padding]);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            padded.set(image.get(y, x), y + padding, x + padding);
        }
    }

    const outH = Math.floor((padded.shape[0] - kh) / stride) + 1;
    const outW = Math.floor((padded.shape[1] - kw) / stride) + 1;
    const output = Tensor.zeros([outH, outW]);

    for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
            let sum = 0;
            for (let ky = 0; ky < kh; ky++) {
                for (let kx = 0; kx < kw; kx++) {
                    const iy = oy * stride + ky;
                    const ix = ox * stride + kx;
                    sum += padded.get(iy, ix) * kernel.get(ky, kx);
                }
            }
            output.set(sum, oy, ox);
        }
    }

    return output;
}

function sliceChannel(input: Tensor, channelIndex: number): Tensor {
    const [channels, h, w] = input.shape as [number, number, number];
    if (channelIndex < 0 || channelIndex >= channels) {
        throw new Error(`Channel index out of range: ${channelIndex}`);
    }

    const channel = Tensor.zeros([h, w]);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            channel.set(input.get(channelIndex, y, x), y, x);
        }
    }
    return channel;
}

function writeChannel(output: Tensor, channelIndex: number, channel: Tensor): void {
    const [h, w] = channel.shape as [number, number];
    const [channels] = output.shape as [number];
    if (channelIndex < 0 || channelIndex >= channels) {
        throw new Error(`Channel index out of range: ${channelIndex}`);
    }

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            output.set(channel.get(y, x), channelIndex, y, x);
        }
    }
}

function flip2d(kernel: Tensor): Tensor {
    const [h, w] = kernel.shape as [number, number];
    const flipped = Tensor.zeros([h, w]);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            flipped.set(kernel.get(y, x), h - 1 - y, w - 1 - x);
        }
    }

    return flipped;
}

function pad2d(image: Tensor, pad: number): Tensor {
    const [h, w] = image.shape as [number, number];
    const padded = Tensor.zeros([h + 2 * pad, w + 2 * pad]);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            padded.set(image.get(y, x), y + pad, x + pad);
        }
    }

    return padded;
}

export class Conv2D {
    inChannels: number;
    outChannels: number;
    kernelSize: number;
    stride: number;
    padding: number;

    kernels: Tensor[][];
    biases: Tensor[];

    gradKernels: Tensor[][] | null = null;
    gradBiases: Tensor[] | null = null;
    private lastInput: Tensor | null = null;

    constructor(
        inChannels: number,
        outChannels: number,
        kernelSize: number,
        options: Conv2DOptions = {},
    ) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = options.stride ?? 1;
        this.padding = options.padding ?? 0;

        let seed = options.seed ?? 42;
        this.kernels = [];
        for (let o = 0; o < outChannels; o++) {
            const row: Tensor[] = [];
            for (let c = 0; c < inChannels; c++) {
                row.push(randomKernel(kernelSize, seed++));
            }
            this.kernels.push(row);
        }

        this.biases = [];
        for (let o = 0; o < outChannels; o++) {
            this.biases.push(new Tensor(new Float64Array([0]), [1]));
        }
    }

    forward(input: Tensor): Tensor {
        if (input.shape.length !== 3) {
            throw new Error("Conv2D.forward expects shape [C, H, W].");
        }
        if (input.shape[0] !== this.inChannels) {
            throw new Error(`Expected ${this.inChannels} channels, got ${input.shape[0]}.`);
        }

        this.lastInput = input;
        const [, h, w] = input.shape as [number, number, number];
        const outH = Math.floor((h + 2 * this.padding - this.kernelSize) / this.stride) + 1;
        const outW = Math.floor((w + 2 * this.padding - this.kernelSize) / this.stride) + 1;

        const output = Tensor.zeros([this.outChannels, outH, outW]);

        for (let o = 0; o < this.outChannels; o++) {
            const acc = Tensor.zeros([outH, outW]);
            for (let c = 0; c < this.inChannels; c++) {
                const channel = sliceChannel(input, c);
                const conv = convolve2d(channel, this.kernels[o]![c]!, {
                    stride: this.stride,
                    padding: this.padding,
                });

                for (let i = 0; i < acc.data.length; i++) {
                    acc.data[i]! += conv.data[i]!;
                }
            }

            for (let i = 0; i < outH; i++) {
                for (let j = 0; j < outW; j++) {
                    output.set(acc.get(i, j) + this.biases[o]!.data[0]!, o, i, j);
                }
            }
        }

        return output;
    }

    backward(gradOutput: Tensor): Tensor {
        if (this.lastInput === null) {
            throw new Error("Must call forward() before backward().");
        }
        if (this.stride !== 1) {
            throw new Error("Conv2D.backward educational implementation supports stride=1 only.");
        }

        const input = this.lastInput;
        const [, h, w] = input.shape as [number, number, number];
        const k = this.kernelSize;

        this.gradKernels = [];
        this.gradBiases = [];
        const gradInput = Tensor.zeros([this.inChannels, h, w]);

        for (let o = 0; o < this.outChannels; o++) {
            const gradOutChannel = sliceChannel(gradOutput, o);

            let gb = 0;
            for (const v of gradOutChannel.data) gb += v;
            this.gradBiases.push(new Tensor(new Float64Array([gb]), [1]));

            const gradRow: Tensor[] = [];
            for (let c = 0; c < this.inChannels; c++) {
                const inChannel = sliceChannel(input, c);
                const gK = Tensor.zeros([k, k]);

                const padded = pad2d(inChannel, this.padding);
                const [outH, outW] = gradOutChannel.shape as [number, number];

                for (let a = 0; a < k; a++) {
                    for (let b = 0; b < k; b++) {
                        let sum = 0;
                        for (let i = 0; i < outH; i++) {
                            for (let j = 0; j < outW; j++) {
                                sum += padded.get(i + a, j + b) * gradOutChannel.get(i, j);
                            }
                        }
                        gK.set(sum, a, b);
                    }
                }
                gradRow.push(gK);

                const kernel = this.kernels[o]![c]!;
                const flipped = flip2d(kernel);
                const contrib = convolve2d(
                    pad2d(gradOutChannel, k - 1 - this.padding),
                    flipped,
                    { stride: 1, padding: 0 },
                );

                const gIn = sliceChannel(gradInput, c);
                for (let i = 0; i < h; i++) {
                    for (let j = 0; j < w; j++) {
                        if (i < contrib.shape[0]! && j < contrib.shape[1]!) {
                            gIn.set(gIn.get(i, j) + contrib.get(i, j), i, j);
                        }
                    }
                }
                writeChannel(gradInput, c, gIn);
            }
            this.gradKernels.push(gradRow);
        }

        return gradInput;
    }

    parameters(): Parameter[] {
        const params: Parameter[] = [];

        for (let o = 0; o < this.outChannels; o++) {
            for (let c = 0; c < this.inChannels; c++) {
                const kernel = this.kernels[o]![c]!;
                params.push({
                    name: `conv.k[${o},${c}]`,
                    data: kernel,
                    grad: this.gradKernels?.[o]?.[c] ?? Tensor.zeros(kernel.shape),
                });
            }
            const bias = this.biases[o]!;
            params.push({
                name: `conv.b[${o}]`,
                data: bias,
                grad: this.gradBiases?.[o] ?? Tensor.zeros(bias.shape),
            });
        }

        return params;
    }
}
