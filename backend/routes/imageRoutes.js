const express = require("express");
const router = express.Router();

const upload = require("../utils/multerConfig");
const { uploadImage } = require("../controllers/imageController");

const Image = require("../models/Image");

router.post("/upload", upload.single("image"), uploadImage);
router.get("/history/:userId", async (req, res) => {
  try {
    const images = await Image.find({ userId: req.params.userId });

    res.json(images);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;