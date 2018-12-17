import 'bootstrap/dist/css/bootstrap.css'; //making use of Bootstrap CSS classes
import * as tf from '@tensorflow/tfjs'; //making use of Tensorflow library

import {MnistData} from './data';
import { deflateRaw } from 'zlib';

var model;
//function allowing for text messages to be placed in Log Output
function createLogEntry(entry) {
    document.getElementById('log').innerHTML += '<br>' + entry;
}
//building the learning model
function createModel() {
    createLogEntry('Creating the model ...');
    model = tf.sequential();
    createLogEntry('Model created');
    //initial 2D layer
    createLogEntry('Adding layers ...');
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1], //size of pixel images
        kernelSize: 5, //flitering the image
        filters: 8, 
        strides: 1,
        activation: 'relu', //Reactified Linear Unit
        kernelInitializer: 'VarianceScaling'
    }));
    //downsizing the input images, gotten from first layer
    model.add(tf.layers.maxPooling2d({
        poolSize: [2,2],
        strides: [2,2]
    }));
    //3rd layer
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));
    //4th layer
    model.add(tf.layers.maxPooling2d({
        poolSize: [2,2],
        strides: [2,2]
    }));
    //flattening output from previous layer
    model.add(tf.layers.flatten());
    //dense layer - performing final classification of an image
    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax' //creating a probability distribution of the classes
    }));

    createLogEntry('Layers have been created');

    createLogEntry('Model is compiling ...');
    model.compile({
        optimizer: tf.train.sgd(0.15), //defining Stochastic learning method 
        loss: 'categoricalCrossentropy'
    });
    createLogEntry('Compiled');
}

let data; //getting access to data
async function load() {
    createLogEntry('Loading MNIST data ...');
    data = new MnistData();
    await data.load(); //loading data from remote location 
    createLogEntry('Data has been loaded successfully');
}
//model training parameters
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;
//model training
async function train() {
    createLogEntry('Starting the training ...');
    for (let i = 0; i < TRAIN_BATCHES; i++) {
        const batch = tf.tidy(() => {
            const batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
            return batch;
        });

        await model.fit(
            batch.xs, batch.labels, {batchSize: BATCH_SIZE, epochs: 1}
        );

        tf.dispose(batch);

        await tf.nextFrame();
    }
    createLogEntry('Training is now completed');
}
//listing functions to be executed in the right order
async function main() {
    createModel();
    await load();
    await train();
    document.getElementById('selectTestDataButton').disabled = false;
    document.getElementById('selectTestDataButton').innerText = "Ramdom data selection and Recognition process";
}

async function predict(batch) {
    tf.tidy(() => {
        const input_value = Array.from(batch.labels.argMax(1).dataSync()); //extracting the original value

        const div = document.createElement('div'); //prediction output element
        div.className = 'prediction-div'; 

        const output = model.predict(batch.xs.reshape([-1, 28, 28, 1])); //passing image data

        const prediction_value = Array.from(output.argMax(1).dataSync()); //extracting prediction probability value
        const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]]); //storing the chosen image

        const canvas = document.createElement('canvas'); //image output element
        canvas.className = 'prediction-canvas';
        draw(image.flatten(), canvas);

        const label = document.createElement('div'); //label output elements
        label.innerHTML = 'Original Value: ' + input_value;
        label.innerHTML += '<br>Prediction Value: ' + prediction_value;
        //checking whether prediction value equals the input value
        if (prediction_value - input_value == 0) { 
            label.innerHTML += '<br>Value recognized successfully!';
        } else {
            label.innerHTML += '<br>Recognition failed!'
        }

        div.appendChild(canvas);
        div.appendChild(label);
        document.getElementById('predictionResult').appendChild(div);
    });
}
//outputing the image through canvas
function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}
//selecting patterns to offset the prediction
document.getElementById('selectTestDataButton').addEventListener('click', async (el,ev) => {
    const batch = data.nextTestBatch(1);
    await predict(batch);
});

main();