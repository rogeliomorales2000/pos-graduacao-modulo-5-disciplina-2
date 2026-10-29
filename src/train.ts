import { pathToFileURL } from "node:url";
import { ImageDataset } from "./dataset";
import { loadImageAsTensor } from "./imagem-loader";
import { defaultCnnConfig, TinyCNN } from "./tiny-cnn";
import trainingLoop from "./training-loop";
import { saveModel } from "./io";

export async function trainCatvsdog(
   options:{
    datasetRoot: string,
    epochs ?: number ,
    learningRate ?: number,
    imagemSize ?: number,
    modelPath ?: string}
) {
    const dataset = new ImageDataset(options.datasetRoot, options.imagemSize ?? 32)
    const train = dataset.list("train").samples
    const val = dataset.list("validation").samples

    const config = defaultCnnConfig();
    const model = new TinyCNN(config);
    await trainingLoop.trainCnn(model,dataset,train,val,{
        epochs: options.epochs ?? 1,
        learningRate: options.learningRate ?? 0.001

    })
    const test = dataset.list("test").samples
    if (
        test.length > 0
    ){
        const testStatus=await trainingLoop.evaluate(model,dataset,test)
        console.log(`teste CC ${(testStatus.accuracy * 100).toFixed(1)}%`)
        if (options.modelPath){
            saveModel(model,options.modelPath)
        }


    }
    return model 
}

/** Classifica uma imagem em "cats" ou "dogs" usando um modelo já treinado. */
export async function predictCatVsDog(
    model: TinyCNN,
    imagePath: string,
    imageSize: number = model.config.inputHeight,
): Promise<{ className: "cats" | "dogs"; dogProbability: number }> {
    const image = await loadImageAsTensor(imagePath, imageSize);
    const prediction = model.forward(image);
    const dogProbability = prediction.get(0, 0);
    return {
        className: dogProbability >= 0.5 ? "dogs" : "cats",
        dogProbability,
    };
}

async function main() {
    const datasetRoot = process.argv[2] ?? "./dataset";
    const epochs = Number(process.argv[3] ?? 1);
    const learningRate = Number(process.argv[4] ?? 0.001);
    const modelPath = process.argv[5] ?? "./model.json";

    await trainCatvsdog({
        datasetRoot,
        epochs,
        learningRate,
        imagemSize: 32,
        modelPath,
    });
}

const isDirectExecution =
    typeof process.argv[1] === "string" &&
    import.meta.url === pathToFileURL(process.argv[1]).href;

if (isDirectExecution) {
    await main();
}