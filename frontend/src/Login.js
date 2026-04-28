import React, { useState } from "react";
import axios from "axios";
import "./Auth.css";

function Login({ setPage }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);

  const handleLogin = async () => {
    try {
      const res = await axios.post(`${process.env.REACT_APP_API_URL}/api/auth/login`, {
        email,
        password,
      });

      console.log("LOGIN RESPONSE:", res.data);


      if (!res.data || !res.data.token) {
        alert("Invalid login response");
        return;
      }

      
      localStorage.setItem("token", res.data.token);

      
      localStorage.setItem(
        "userId",
        res.data.userId ? res.data.userId : ""
      );

      
      localStorage.setItem(
        "role",
        res.data.role ? res.data.role : "user"
      );

      
      if (res.data.name) {
        localStorage.setItem("name", res.data.name);
      }

      console.log("Stored role:", localStorage.getItem("role"));

      alert("Login successful");

      
      setPage("home");

    } catch (err) {
      console.log("LOGIN ERROR:", err.response?.data || err.message);
      alert("Login failed");
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <h2 className="auth-title">🔐 Login</h2>

        <input
          type="email"
          placeholder="Email"
          className="auth-input"
          onChange={(e) => setEmail(e.target.value)}
        />

        {/* PASSWORD */}
        <div className="password-wrapper">
          <input
            type={show ? "text" : "password"}
            placeholder="Password"
            className="auth-input"
            onChange={(e) => setPassword(e.target.value)}
          />

          <span className="eye-icon" onClick={() => setShow(!show)}>
            {show ? "🙈" : "👁️"}
          </span>
        </div>

        <button className="auth-btn" onClick={handleLogin}>
          Login
        </button>

        <p className="auth-link" onClick={() => setPage("signup")}>
          Don't have an account? Signup
        </p>
      </div>
    </div>
  );
}

export default Login;