'use strict';

let PRloss = 0
let frames = 0

class PolynomialRegression {
  constructor(color, grade) {
    this.color = color
    this.coef = new Array(grade+1).fill()
    for (let i = 0; i < this.coef.length; i++) {
      this.coef[i] = tf.variable(tf.scalar(0))
    }
    this.loss = (pred, labels) => pred.sub(labels).square().mean()
    this.optimizer = tf.train.adam(0.2)
  }

  f(x_vals) {
    const xs = tf.tensor1d(x_vals)
    let ys = tf.zerosLike(xs)
    for (let i = 0; i < this.coef.length; i++)
      ys = ys.add(xs.pow(tf.scalar(i)).mul(this.coef[this.coef.length-i-1]))
    return ys
  }

  getLoss() {
    return this.loss(
      this.f(points.map(p => p.x)),
      tf.tensor1d(points.map(p => p.y))
    )
  }

  getLossValue() {
    if (points.length < 2) return 0
    return tf.tidy(() => this.getLoss().dataSync())[0]
  }

  update(points) {
    if (points.length < 2 || mouseIsPressed) return this
    tf.tidy(() => this.optimizer.minimize(() => this.getLoss()))
    return this
  }

  draw() {
    let xs = []
    for (let i = 0; i <= width; i++) {
      xs.push(i)
    }
    xs = xs.map(x => map(x, 0, width, 0, 1))
    const ys = tf.tidy(() => this.f(xs).dataSync())

    beginShape()
    noFill()
    stroke('#2980b9')
    strokeWeight(2)
    for (let i = 0; i <= width; i++) {
      let x = map(xs[i], 0, 1, 0, width)
      let y = map(ys[i], 1, 0, 0, width)
      vertex(x, y)
    }
    endShape()
  }
}

const points = []
// Polynomial Regression
const PR = new PolynomialRegression('#2980b9', 5)

function setup() {
  createCenteredCanvas()
  frameRate(60)
  draw()
}

function windowResized() {
  createCenteredCanvas()
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
  line(width-140, 15, width-130, 15)
  // Legend text
  noStroke()
  fill('#fff')
  textAlign(LEFT, CENTER)
  textSize(10)
  text('Polynomial Regression', width-120, 15)
  text('Loss: ' + PRloss, width-120, 30)
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