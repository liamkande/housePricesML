require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
    return features
        .sub(predictionPoint)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels.reshape([labels.shape[0], 1]), 1) // Reshape labels to match the desired shape
        .unstack()
        .sort((a, b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1) // Use arraySync() to access the tensor values
        .slice(0, k)
        .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k; // Use arraySync() to access the tensor values
}

let { features, labels, testFeatures, testLabels } = loadCSV("kc_house_data.csv", {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long"],
    labelColumns: ["price"]
});

labels = tf.tensor(labels); // Convert labels to a tensor

console.log(testFeatures);
const result = knn(tf.tensor(features), labels, tf.tensor(testFeatures[0]), 10);
console.log("Guess:", result, "Actual:", testLabels[0][0]);
