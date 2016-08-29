// example is based on the numpy neural network tutorial featured here
// https://iamtrask.github.io/2015/07/12/basic-python-network/
// 'use strict';

var Matrix = require('vectorious').Matrix;

function sigmoid(ddx) {
  return function (x) {
    return ddx ?
      x * (1 - x) :
      1.0 / (1 + Math.exp(-x));
  };
}

// input
var X = new Matrix([
  [0, 0, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0],
  [0, 0, 1, 1],
  [0, 1, 0, 0],
  [0, 1, 1, 0],
  [0, 1, 0, 1],
  [0, 1, 1, 1],
  [1, 0, 0, 0],
  [1, 0, 0, 1],
  [1, 0, 1, 0],
]);

// output
var y = new Matrix([ [0, 1],
  [1, 0], [0, 1],
  [1, 0], [0, 1],
  [1, 0], [0, 1],
  [1, 0], [0, 1],
  [1, 0], [0, 1]
]);
// initialize weights with a standard deviation of 2 and mean -1
var syn0 = Matrix.random.apply(null, [4,6], 2, -1),
    syn1 = Matrix.random.apply(null, [6,2], 2, -1);

// layers and deltas
var l0, l1, l0_delta, l1_delta;

for (var i = 0; i < 500; i++) {
  l0 = X.multiply(syn0).map(sigmoid());
  l1 = l0.multiply(syn1).map(sigmoid());
  // console.log(l0.shape, l1.shape);

  l1_delta = Matrix.subtract(y, l1).product(l1.map(sigmoid(true)));
  // console.log(l1_delta.shape);
  l0_delta = l1_delta.multiply(syn1.T).product(l0.map(sigmoid(true)));
  // console.log(l0_delta.shape);

  syn1.add(l0.T.multiply(l1_delta));
  syn0.add(X.T.multiply(l0_delta));
  // console.log(i)
}

// final trained neural network output!
// should be close to [[0, 1, 1, 0]] transpose
// console.log(l1.toArray());

var predict = new Matrix([[1, 1, 1, 1]]).multiply(syn0).map(sigmoid());
var ans = predict.multiply(syn1).map(sigmoid());
console.log(ans.toArray()[0])
