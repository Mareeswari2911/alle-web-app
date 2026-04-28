import React, { useState } from "react";
import axios from "axios";
import "./Auth.css";
import logo from "./assets/logo1.png";

function Signup({ setPage }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);

  const handleSignup = async () => {
    try {
      await axios.post(`${process.env.REACT_APP_API_URL}/api/auth/register`, {
        email,
        password,
      });

      alert("Signup successful");
      setPage("login");
    } catch (err) {
      alert("Signup failed");
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-wrapper">

        {/* ✅ BIG LOGO */}
        <div className="logo-floating">
          <img src={logo} alt="App Logo" className="main-logo" />
        </div>

        {/* ✅ SIGNUP BOX */}
        <div className="auth-box">
          <h2 className="auth-title">Signup</h2>

          <input
            type="email"
            placeholder="Email"
            className="auth-input"
            onChange={(e) => setEmail(e.target.value)}
          />

          <div className="password-wrapper">
            <input
              type={show ? "text" : "password"}
              placeholder="Password"
              className="auth-input"
              onChange={(e) => setPassword(e.target.value)}
            />
            <span
              className="eye-icon"
              onClick={() => setShow(!show)}
            >
              {show ? "🙈" : "👁️"}
            </span>
          </div>

          <button className="auth-btn" onClick={handleSignup}>
            Signup
          </button>

          <p className="auth-link" onClick={() => setPage("login")}>
            Already have an account? Login
          </p>
        </div>

      </div>
    </div>
  );
}

export default Signup;