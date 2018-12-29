require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
});

const lr = new LinearRegression(features, labels, {
    learningRate: 0.001,
    iterations: 100
});

lr.train();

console.error('updated m is ', lr.m, ' Updated B is ', lr.b);