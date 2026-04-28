import React, { useState } from "react";
import Login from "./Login";
import Signup from "./Signup";
import Home from "./Home";
import History from "./History";
import Admin from "./Admin"; 

function App() {
  const [page, setPage] = useState("signup");

  const token = localStorage.getItem("token");
  const role = localStorage.getItem("role"); 

  
  if (!token) {
    return (
      <>
        {page === "login" && <Login setPage={setPage} />}
        {page === "signup" && <Signup setPage={setPage} />}
      </>
    );
  }

  
  return (
    <>
      {/* ADMIN PAGE */}
      {role === "admin" && page === "admin" && (
        <Admin setPage={setPage} />
      )}

      {/* USER PAGES */}
      {page === "home" && <Home setPage={setPage} />}
      {page === "history" && <History setPage={setPage} />}

      {/* DEFAULT ROUTE */}
      {page !== "home" &&
        page !== "history" &&
        page !== "admin" &&
        <Home setPage={setPage} />
      }

      {/* ADMIN QUICK ACCESS (optional debug button) */}
      {role === "admin" && page === "home" && (
        <button
          style={{
            position: "fixed",
            bottom: 20,
            right: 20,
            padding: "10px",
            background: "black",
            color: "white",
            borderRadius: "8px"
          }}
          onClick={() => setPage("admin")}
        >
          Admin Panel
        </button>
      )}
    </>
  );
}

export default App;