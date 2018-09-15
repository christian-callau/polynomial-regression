'use strict';

class PolynomialRegression {
  constructor(color, grade, learningRate, beta1) {
    this.color = color
    this.coef = new Array(grade + 1).fill()
      .map(() => tf.variable(tf.scalar(Math.random())))
    this.loss = (pred, labels) => pred.sub(labels).square().mean()
    this.optimizer = tf.train.adam(learningRate, beta1)
    this.name = 'grade: '+grade+' rate: '+learningRate+' beta1: '+beta1
  }

  f(x_vals) {
    const xs = tf.tensor1d(x_vals)
    let ys = tf.zerosLike(xs)
    for (let i = 0; i < this.coef.length; i++)
      ys = ys.add(xs.pow(tf.scalar(i)).mul(this.coef[i]))
    return ys
  }

  getLoss() {
    return this.loss(
      this.f(points.map(p => p.x)),
      tf.tensor1d(points.map(p => p.y))
    )
  }

  getLossValue() {
    if (points.length < 1) return this
    return tf.tidy(() => this.getLoss().dataSync())[0]
  }

  getName() {
    return this.name
  }

  update(points) {
    if (points.length < 1 || mouseIsPressed) return this
    tf.tidy(() => this.optimizer.minimize(() => this.getLoss()))
    return this
  }

  draw() {
    const xs = [...Array(width + 1).keys()].map(x => map(x, 0, width, 0, 1))
    const ys = tf.tidy(() => this.f(xs).dataSync())
    beginShape()
    noFill()
    stroke('#2980b9')
    strokeWeight(2)
    for (let i = 0; i <= width; i++)
      vertex(map(xs[i], 0, 1, 0, width), map(ys[i], 1, 0, 0, width))
    endShape()
  }
}

const points = []
const PR = new PolynomialRegression('#2980b9', 7, .1, .8)
let PRloss = 0
let frames = 0

function setup() {
  createCenteredCanvas()
  frameRate(60)
  draw()
}

function windowResized() {
  createCenteredCanvas()
  draw()
}

function createCenteredCanvas() {
  const size =
    (windowWidth <= 540 || windowHeight <= 630) +
    (windowWidth <= 960 || windowHeight <= 800)
  const dim = [600, 450, 300][size]
  createCanvas(dim, dim).position(
    (windowWidth - width) / 2,
    (windowHeight - height) / 2
  )
}

function draw() {
  background('#0e0e0e')
  PR.update(points).draw()
  if (frames++ === 60) {
    PRloss = Math.round(PR.getLossValue() * 10 ** 8) / 10 ** 8
    frames = 0
  }
  drawPoints()
  drawFrame()
}

function drawPoints() {
  stroke('#fff')
  strokeWeight(3)
  points.map(p => point(
    map(p.x, 0, 1, 0, width),
    map(p.y, 0, 1, height, 0)
  ))
}

function drawFrame() {
  // Frame
  stroke('#000')
  strokeWeight(1)
  line(0, 0, width, 0)
  line(0, 0, 0, height)
  line(width-1, height-1, width-1, 0)
  line(width-1, height-1, 0, height-1)
  // Legend lines
  strokeWeight(2)
  stroke(PR.color)
  line(width-160, 15, width-150, 15)
  // Legend text
  noStroke()
  fill('#fff')
  textAlign(LEFT, CENTER)
  textSize(10)
  text(PR.getName(), width-140, 15)
  text('loss: ' + PRloss, width-120, 30)
}

function mousePressed() {
  if (mouseX > 0 && mouseX < width &&
      mouseY > 0 && mouseY < height) {
    points.push({
      x: map(mouseX, 0, width, 0, 1),
      y: map(mouseY, 0, height, 1, 0)
    })
  }
}
