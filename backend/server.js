const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();


app.use(cors({
  origin: [
    'http://localhost:3000',
    'https://alle-web-app.vercel.app'
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json());

const dotenv = require("dotenv");
dotenv.config();


app.use("/uploads", express.static("uploads"));


const PORT = process.env.PORT || 5000;


mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("MongoDB Connected"))
  .catch(err => {
    console.error(" DB Error:", err);
    process.exit(1);
  });


app.get("/", (req, res) => {
  res.send("Backend Running...");
});


const authRoutes = require("./routes/authRoutes");
app.use("/api/auth", authRoutes);


const imageRoutes = require("./routes/imageRoutes");
app.use("/api/image", imageRoutes);


const authMiddleware = require("./middleware/authMiddleware");
app.get("/api/protected", authMiddleware, (req, res) => {
  res.json({
    message: "Protected route accessed",
    user: req.user
  });
});

const enhanceRoutes = require("./routes/enhanceRoutes");


app.use("/api/enhance", enhanceRoutes);


const adminRoutes = require("./routes/adminRoutes");

app.use("/api/admin", adminRoutes);



app.use((err, req, res, next) => {
  console.error(" Error:", err.message);
  res.status(500).json({ error: err.message || "Something went wrong" });
});


app.listen(PORT, () => {
  console.log(` Server running on http://localhost:${PORT}`);
});