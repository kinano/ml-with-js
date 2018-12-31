require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg']
});

const lr = new LinearRegression(features, labels, {
    learningRate: 10,
    iterations: 100
});

lr.train();
const r2 = lr.test(testFeatures, testLabels);
console.error('R2 is', r2);