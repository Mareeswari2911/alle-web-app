import React, { useEffect, useState, useCallback } from "react";
import axios from "axios";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";

import { Bar } from "react-chartjs-2";
import "./Admin.css";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function Admin({ setPage }) {
  const [users, setUsers] = useState([]);
  const [ranking, setRanking] = useState([]);
  const [analytics, setAnalytics] = useState([]);

  const token = localStorage.getItem("token");

  
  const fetchUsers = useCallback(async () => {
    try {
      const res = await axios.get("http://localhost:5000/api/admin/users", {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUsers(res.data);
    } catch (err) {
      console.error("Error fetching users:", err);
    }
  }, [token]);

  const deleteUser = async (id) => {
    try {
      await axios.delete(`http://localhost:5000/api/admin/users/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchUsers();
    } catch (err) {
      console.error("Error deleting user:", err);
    }
  };

  
  const fetchRanking = useCallback(async () => {
    try {
      const res = await axios.get("http://localhost:5000/api/admin/ranking", {
        headers: { Authorization: `Bearer ${token}` }
      });
      setRanking(res.data);
    } catch (err) {
      console.error("Error fetching ranking:", err);
    }
  }, [token]);

  
  const fetchAnalytics = useCallback(async () => {
    try {
      const res = await axios.get(
        "http://localhost:5000/api/admin/analytics/uploads",
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setAnalytics(res.data);
    } catch (err) {
      console.error("Error fetching analytics:", err);
    }
  }, [token]);

  useEffect(() => {
  
  fetchUsers();
  fetchRanking();
  fetchAnalytics();

  
  const interval = setInterval(() => {
    fetchUsers();
    fetchRanking();
    fetchAnalytics();
  }, 5000);

  
  return () => clearInterval(interval);

}, [fetchUsers, fetchRanking, fetchAnalytics]);
  
  const uploadsChart = {
    labels: analytics.map((a) => a._id),
    datasets: [
      {
        label: "Uploads per Day",
        data: analytics.map((a) => a.count),
        backgroundColor: "#4f46e5"
      }
    ]
  };

  const qualityChart = {
    labels: ranking.slice(0, 5).map((_, i) => `Image ${i + 1}`),
    datasets: [
      {
        label: "PSNR Score",
        data: ranking.slice(0, 5).map((img) => img.psnr),
        backgroundColor: "#10b981"
      }
    ]
  };

  return (
    <div className="admin-container">

      {/* HEADER */}
      <div className="admin-header">
        <h1>🛠 Admin Dashboard</h1>

        <button className="back-btn" onClick={() => setPage("home")}>
          ⬅ Back
        </button>
      </div>

      {/* CARDS */}
      <div className="card-row">
        <div className="card">
          <h3>👥 Users</h3>
          <h2>{users.length}</h2>
        </div>

        <div className="card">
          <h3>📸 Images</h3>
          <h2>{ranking.length}</h2>
        </div>

        <div className="card">
          <h3>📊 Upload Days</h3>
          <h2>{analytics.length}</h2>
        </div>
      </div>

      {/* CHARTS */}
      <div className="chart-row">

        <div className="chart-box">
          <h3>📊 Upload Activity</h3>
          <Bar data={uploadsChart} />
        </div>

        <div className="chart-box">
          <h3>🏆 Image Quality Score</h3>
          <Bar data={qualityChart} />
        </div>

      </div>

      {/* USERS */}
      <h2 className="section-title">👥 User Management</h2>

      <div className="user-list">
        {users.map((user) => (
          <div key={user._id} className="user-card">
            <div>
              <p className="name">{user.name}</p>
              <p className="email">{user.email}</p>
              <span className="role">{user.role}</span>
            </div>

            <button
              className="delete-btn"
              onClick={() => deleteUser(user._id)}
            >
              Delete
            </button>
          </div>
        ))}
      </div>

      {/* IMAGE RANKING */}
      <h2 className="section-title">🏆 Top Quality Images</h2>

      <div className="grid">
        {ranking.slice(0, 6).map((img, i) => (
          <div key={i} className="image-card">
            <img src={img.enhancedImage} alt="enhanced" />
            <p>PSNR: {img.psnr}</p>
            <p>SSIM: {img.ssim}</p>
          </div>
        ))}
      </div>

    </div>
  );
}

export default Admin;