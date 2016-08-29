var Matrix = require('vectorious').Matrix;

function Network({inputLayer, hiddenLayers, outputLayer}) {
  this.inputLayer = inputLayer;
  this.hiddenLayers = hiddenLayers;
  this.outputLayer = outputLayer;
  this.weightMatricies = this.createWeights([inputLayer, ...hiddenLayers, outputLayer])
  this.guesses = this.weightMatricies.map( mat => Matrix.zeros(mat.shape[0], mat.shape[1]))
  this.deltas = this.weightMatricies.map( mat => Matrix.zeros(mat.shape[0], mat.shape[1]))
}
Network.prototype.createWeights = function createWeights(weightArray) {
  const weights = []
  for(var i = 1; i < weightArray.length; i++) {
    const column = weightArray[i]
    const row = weightArray[i - 1]
    weights.push(Matrix.random.apply(null, [row, column], 2, -1))
  }
  return weights
};
Network.prototype.train = function train(inputs, target, opts = {}) {
  var cycles = opt.cycles || 500
  var learningRate = opt.learningRate || 1
  var X = new Matrix(inputs)
  var y = new Matrix(target)

  for (var i = 0; i < cycles; i++) {
    const yHat = this.forward(X);

    this.backPropagate(yHat, y)
    this.updateWeights()
  }
}
Network.prototype.forward = function forward(input) {
  const guesses = [input]
  const finalOutput = this.weightMatricies.reduce( (output, W) => {
    const guess = output.multiply(W).map( x => this.sigmoid(x))
    guesses.push(guess)
    return guess
  }, input)
  this.guesses = guesses;
  return finalOutput;
}

// l1_delta = Matrix.subtract(y, l1).product(l1.map(sigmoid(true)));
// l0_delta = l1_delta.multiply(syn1.T).product(l0.map(sigmoid(true)));

Network.prototype.backPropagate = function backPropagate(guess, target) {
  for (var i = this.deltas.length - 1; i >= 0; i--) {
    if (!this.deltas[i + 1]) {
      this.deltas[i] = Matrix
        .subtract( target, guess )
        .product( guess.map( value => this.sigmoidPrime(value) ) )
    } else {
      this.deltas[i] = this.deltas[i + 1]
        .multiply( this.weightMatricies[i + 1].T )
        .product( this.guesses[i + 1].map( value => this.sigmoidPrime(value) ) )
    }
  }
};
// syn1.add(l0.T.multiply(l1_delta));
// syn0.add(X.T.multiply(l0_delta));
Network.prototype.updateWeights = function updateWeights(X) {
  this.weightMatricies.map( (weight, index) => (
    weight.add(this.guesses[index].T.multiply(this.deltas[index]))
  ))
}
Network.prototype.sigmoid = function sigmoid(x) {
  return 1.0 / (1 + Math.exp(-x));
};

Network.prototype.sigmoidPrime = function sigmoidPrime(x) {
  return this.sigmoid(x) * (1 - this.sigmoid(x))
};
Network.prototype.predict = function predict(data) {
  return this.weightMatricies.reduce( (output, W) => {
    return output.multiply(W).map( x => this.sigmoid(x))
  }, new Matrix([data]) ).toArray()[0]
}
module.exports = Network
