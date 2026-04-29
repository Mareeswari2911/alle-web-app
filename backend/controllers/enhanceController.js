const axios = require("axios");
const FormData = require("form-data");
const Image = require("../models/Image");
const sharp = require("sharp");

const {
  calculatePSNR,
  calculateSSIM
} = require("../utils/imageMetrics");

exports.enhanceImage = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const { userId } = req.body;
    if (!userId) {
      return res.status(400).json({ error: "UserId missing" });
    }

    const originalBuffer = req.file.buffer;

    
    const form = new FormData();
    form.append("file", originalBuffer, req.file.originalname);

    const response = await axios.post(
      `${process.env.FASTAPI_URL}/api/enhance`,
        form,
        { headers: form.getHeaders() }
    );

    const data = response.data;

    
    const originalBase64 = data.original_image;
    const enhancedBase64 = data.denoised_image; 

    
    const size = 256;

    const orig = await sharp(originalBuffer)
      .resize(size, size)
      .removeAlpha()
      .raw()
      .toBuffer();

    const enhBuffer = Buffer.from(enhancedBase64, "base64");

    const enh = await sharp(enhBuffer)
      .resize(size, size)
      .removeAlpha()
      .raw()
      .toBuffer();

    let mse = 0;
    for (let i = 0; i < orig.length; i++) {
      mse += (orig[i] - enh[i]) ** 2;
    }
    mse /= orig.length;

    const psnr = calculatePSNR(mse);
    const ssim = calculateSSIM(orig, enh);

    const saved = await Image.create({
      userId,
      originalImage: `data:image/png;base64,${originalBase64}`,
      enhancedImage: `data:image/png;base64,${enhancedBase64}`,
      psnr,
      ssim
    });

    
    res.json({
      image: saved
    });

  } catch (error) {
    console.error("Enhance Error:", error.message);
    res.status(500).json({ error: "Enhancement failed" });
  }
};