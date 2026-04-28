const Image = require("../models/Image");

exports.uploadImage = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    const { userId, psnr, ssim } = req.body;

    if (!userId) {
      return res.status(400).json({ message: "UserId missing" });
    }

    const baseUrl = "http://localhost:5000";

    const fileUrl = `${baseUrl}/${req.file.path.replace(/\\/g, "/")}`;

    const newImage = await Image.create({
      userId,
      originalImage: fileUrl,
      enhancedImage: fileUrl,
      psnr: psnr || null,
      ssim: ssim || null
    });

    res.json({
      message: "Image uploaded successfully",
      image: newImage
    });

  } catch (err) {
    console.log("UPLOAD ERROR:", err);
    res.status(500).json({ error: err.message });
  }
};