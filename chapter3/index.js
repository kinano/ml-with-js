require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg']
});

const lr = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10
});

lr.train();
const r2 = lr.test(testFeatures, testLabels);

plot({
    x: lr.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
});

console.error('R2 is', r2);

lr.predict(
    [
        [130, 307, 1.2]
    ]
).print();