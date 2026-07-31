function clamp(num, min = 1, max = 10) {
  return Math.min(Math.max(num, min), max);
}

module.exports = clamp;