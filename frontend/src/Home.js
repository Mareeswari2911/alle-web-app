import React, { useState } from "react";
import axios from "axios";
import "./Home.css";

function Home({ setPage }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [enhanced, setEnhanced] = useState(null);
  const [loading, setLoading] = useState(false);
  const [slider, setSlider] = useState(50);

  const userId = localStorage.getItem("userId");
  const role = localStorage.getItem("role"); 

  const handleChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setEnhanced(null);
  };

  const handleUpload = async () => {
    if (!image) {
      alert("Please select image");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);
    formData.append("userId", userId);

    try {
      setLoading(true);

      const res = await axios.post(
        "http://localhost:5000/api/enhance",
        formData
      );

      const data = res.data.image;

      setPreview(data.originalImage);

      
      setEnhanced(data.enhancedImage);

      setSlider(50);

    } catch (err) {
      console.error(err);
      alert("Enhancement failed");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("userId");
    localStorage.removeItem("role"); 
    window.location.reload();
  };

  const handleDownload = () => {
    if (!enhanced) return;

    const link = document.createElement("a");
    link.href = enhanced;
    link.download = "enhanced-image.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="container">

      {/* HEADER */}
      <div className="header">
        <h1 className="title">✨ Image Enhancer App</h1>

        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          
          <button className="history-btn" onClick={() => setPage("history")}>
            📁 My History
          </button>

          {/* 👑 ADMIN PANEL BUTTON (ONLY FOR ADMIN) */}
          {role === "admin" && (
            <button
              className="btn"
              style={{
                backgroundColor: "#000",
                color: "#fff"
              }}
              onClick={() => setPage("admin")}
            >
              🛠 Admin Panel
            </button>
          )}

          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>

        </div>
      </div>

      {/* UPLOAD SECTION */}
      <div className="card">
        <input type="file" onChange={handleChange} className="file-input" />

        <button className="btn" onClick={handleUpload}>
          {loading ? "Enhancing..." : "Enhance Image"}
        </button>
      </div>

      {/* IMAGE DISPLAY */}
      <div className="image-container">

        {preview && (
          <div className="image-card">
            <h3>Original Image</h3>
            <img src={preview} alt="original" />
          </div>
        )}

        {enhanced && (
          <div className="image-card">
            <h3>Enhanced Image</h3>
            <img src={enhanced} alt="enhanced" />
          </div>
        )}

      </div>

      {/* SLIDER */}
      {preview && enhanced && (
        <div className="slider-wrapper">

          <h3 className="slider-title">Before / After Comparison</h3>

          <div className="slider-container">

            <img src={preview} className="img before" alt="before" />

            <img
              src={enhanced}
              className="img after"
              alt="after"
              style={{ width: `${slider}%` }}
            />

            <input
              type="range"
              min="0"
              max="100"
              value={slider}
              onChange={(e) => setSlider(e.target.value)}
              className="slider"
            />

          </div>
        </div>
      )}

      {/* DOWNLOAD */}
      {enhanced && (
        <div className="download-container">
          <button onClick={handleDownload} className="download-btn">
            ⬇️ Download Enhanced Image
          </button>
        </div>
      )}

    </div>
  );
}

export default Home;