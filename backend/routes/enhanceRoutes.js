const express = require("express");
const router = express.Router();
const multer = require("multer");

const { enhanceImage } = require("../controllers/enhanceController");

const storage = multer.memoryStorage();
const upload = multer({ storage });

router.post("/", upload.single("file"), enhanceImage);

module.exports = router;