const express = require("express");
const router = express.Router();

const authMiddleware = require("../middleware/authMiddleware");
const adminMiddleware = require("../middleware/adminMiddleware");

const User = require("../models/User");
const Image = require("../models/Image");

router.get("/users", authMiddleware, adminMiddleware, async (req, res) => {
  const users = await User.find({}, "-password");
  res.json(users);
});

router.delete("/users/:id", authMiddleware, adminMiddleware, async (req, res) => {
  await User.findByIdAndDelete(req.params.id);
  res.json({ message: "User deleted" });
});


router.get("/analytics/uploads", authMiddleware, adminMiddleware, async (req, res) => {
  const data = await Image.aggregate([
    {
      $group: {
        _id: { $dateToString: { format: "%Y-%m-%d", date: "$createdAt" } },
        count: { $sum: 1 }
      }
    },
    { $sort: { _id: 1 } }
  ]);

  res.json(data);
});
router.get("/ranking", authMiddleware, adminMiddleware, async (req, res) => {
  const images = await Image.find()
    .sort({ psnr: -1, ssim: -1 })
    .limit(20);

  res.json(images);
});

router.get("/export", authMiddleware, adminMiddleware, async (req, res) => {
  const images = await Image.find();

  res.json({
    total: images.length,
    data: images
  });
});

module.exports = router;