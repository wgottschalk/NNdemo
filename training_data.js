
var inputs = [
  [0, 0, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0],
  [0, 0, 1, 1],
  [0, 1, 0, 0],
  [0, 1, 0, 1],
  [0, 1, 1, 0],
  [0, 1, 1, 1],
  [1, 0, 0, 0],
  [1, 0, 0, 1],
  [1, 0, 1, 0]
];
var output = [
  "even",
  "odd",
  "even",
  "odd",
  "even",
  "odd",
  "even",
  "odd",
  "even",
  "odd",
  "even"
]

function cleanOutput(string) {
  return string === "even" ? [1, 0] : [0, 1];
}

module.exports = {
  inputs: inputs,
  outputs: output.map(cleanOutput)
}
