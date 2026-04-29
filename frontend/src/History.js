import React, { useEffect, useState } from "react";
import axios from "axios";
import "./History.css";

function History({ setPage }) {
  const [images, setImages] = useState([]);

  const userId = localStorage.getItem("userId");

  useEffect(() => {
    if (!userId) return;

    axios.get(`${process.env.REACT_APP_API_URL}/api/image/history/${userId}`)
      .then((res) => setImages(res.data))
      .catch((err) => console.log(err));
  }, [userId]);

  return (
    <div className="history-container">

      {/* HEADER */}
      <div className="history-header">
        <h2>📁 My History</h2>

        <button className="back-btn" onClick={() => setPage("home")}>
          ⬅ Back to Home
        </button>
      </div>

      {/* EMPTY STATE */}
      {images.length === 0 ? (
        <p className="empty-text">No images uploaded yet</p>
      ) : (
        <div className="grid">

          {images.map((img, index) => (
            <div className="card" key={index}>

              <h4>Image {index + 1}</h4>

              <div className="image-box">
                <div>
                  <p>Original</p>
                  <img src={img.originalImage} alt="original" />
                </div>

                <div>
                  <p>Enhanced</p>
                  <img src={img.enhancedImage} alt="enhanced" />
                </div>
              </div>

              {/* ✅ METRICS FIXED */}
              <div className="metrics">
                <span>
                  PSNR: {img.psnr ? img.psnr.toFixed(2) : "0.00"} dB
                </span>

                <span>
                  SSIM: {img.ssim ? img.ssim.toFixed(3) : "0.000"}
                </span>
              </div>

              {/* DOWNLOAD */}
              <a href={img.enhancedImage} download>
                <button className="download-btn">
                  ⬇ Download
                </button>
              </a>

            </div>
          ))}

        </div>
      )}
    </div>
  );
}

export default History;