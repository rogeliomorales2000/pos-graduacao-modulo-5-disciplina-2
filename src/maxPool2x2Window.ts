import Tensor from "./tensor";
import type { Nullable } from "./types.ts";

export function masPool2x2Window(

    window : number [] [] 
): number { 
    let m = -Infinity;

    for (
        const row of window
    ){
        for(
            const v of row
        ){
            if (
                v > m
            ){
                m = v ;
            }
        }


    }
    return m 

}


export class MaxPool2D {
    private lastArgMax: Nullable<{
        c:number,
        i: number,
        j: number
    }[]>| null = null
    private lastInputShape: Nullable<number[]>= null
    constructor(
        private poolSize = 2,
        private stride ?: number
    ) {}


    forward(
        input : Tensor
    ): Tensor {
        const [c,h,w] = input.shape as [number,number,number]

        const stride = this.stride ?? this.poolSize

        const outH = Math.floor((h- this.poolSize) / stride)+1

        const outW = Math.floor ((w - this.poolSize)/stride ) +1
        const outC = Math.floor((c-this.poolSize)/stride)+1

        const output = Tensor.zeros([c,outH,outW])

        this.lastArgMax = []
        this.lastInputShape = [c,h,w]

        for(let ch = 0 ; ch < c ; ch++){
            for(
                let i =0 ; i < outH; i++
            ){
                for (
                    let j = 0 ; j < outW ; j++
                ){
                    let maxVal = -Infinity
                    let maxI = 0
                    let maxJ = 0
                    for (
                        let a = 0 ; a < this.poolSize ; a++
                    ){
                        for (
                            let b = 0 ; b < this.poolSize; b++

                        ){
                            const ri = i *stride + a ;
                            const ci = j *stride+b
                            const v = input.get(ch,ri,ci)

                            if (
                                v > maxVal
                            ){
                                maxVal = v
                                maxI = ri
                                maxJ = ci 
                            }
                        }
                    }
                    output.set(maxVal,ch,i,j)
                    this.lastArgMax.push({
                        c : ch,
                        i: maxI,
                        j: maxJ
                    })
                }

            }
        }
        return output
    }

    backward (
        gradOutput : Tensor
    ):Tensor   {
        const gradInput = Tensor.zeros(this.lastInputShape!)
        let idx = 0
        const [c, outH,outW ] = gradOutput.shape as [number,number,number]
        for (
            let ch = 0;ch< c ; ch ++
        ){
            for(
                let i = 0 ; i < outH ; i++
            ){
                for(
                    let j = 0 ; j < outW;j++
                ){
                    const arg = this.lastArgMax![idx++]
                    const prev = gradInput.get(arg.c,arg.i,arg.j)
                    gradInput.set(prev+gradOutput.get(ch,i,j,),arg.c,arg.i,arg.j)

                }
            }
        }
        return gradInput
    }
}
