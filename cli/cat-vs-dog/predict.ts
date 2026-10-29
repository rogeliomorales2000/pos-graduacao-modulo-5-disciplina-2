import { loadModel } from "../../src/io";
import { predictCatVsDog } from "../../src/train";

const model = loadModel("model/cats-vs-dog.json");
const imagePath = process.argv[2];

if (!imagePath) {
    throw new Error("Informe o caminho da imagem para prever: npx tsx cli/predict.ts <imagem>");
}

const result = await predictCatVsDog(model, imagePath, 32);
console.log(result);

// const model = loadModel("models/cat-vs-dog.json");
