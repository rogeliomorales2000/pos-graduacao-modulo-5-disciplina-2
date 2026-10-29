import { pathToFileURL } from "node:url";
import { ImageDataset } from "../../src/dataset";
import { defaultCnnConfig, TinyCNN } from "../../src/tiny-cnn";
import trainingLoop from "../../src/training-loop";
import { saveModel } from "../../src/io";

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