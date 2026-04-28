const mongoose = require("mongoose");

const imageSchema = new mongoose.Schema({
  userId: String,
  originalImage: String,
  enhancedImage: String,
  psnr: Number,
  ssim: Number,
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("Image", imageSchema);