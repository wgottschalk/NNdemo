const {inputs, outputs} = require('./training_data')
const NeuralNetwork = require('./network');

const network = new NeuralNetwork({
  inputLayer: 4,
  hiddenLayers: [10],
  outputLayer: 2
})

network.train(inputs, outputs, {cycles: 50, trainingRate: 0.01})
const x = network.predict([1,1,1,1])
console.log(x)
