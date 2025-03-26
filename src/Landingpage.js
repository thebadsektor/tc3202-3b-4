import React, { useState } from "react";
import "./Landingpage.css";

function Landingpage({ onLogout }) {  // 🔹 Accept onLogout prop
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="landing-container">
      {/* Header */}
      <header className="landing-header">
        <h2>BotaniSnap-AI</h2>
        <div className="user-menu">
          <span>User</span>
          <div className="user-icon"></div>

          {/* Menu Icon (☰) */}
          <div className="menu-icon" onClick={() => setMenuOpen(!menuOpen)}>
            &#9776;
          </div>

          {/* Dropdown Menu */}
          {menuOpen && (
            <div className="dropdown-menu">
              <ul>
                <li>Profile</li>
                <li>Menu</li>
                <li>Gallery</li>
                <li className="logout" onClick={onLogout}>Log Out</li>  {/* 🔹 Calls logout function */}
              </ul>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <div className="landing-main">
        <div className="main-box">
          <button className="action-button">Use camera</button>
          <p>or</p>
          <button className="action-button">Upload a photo</button>
        </div>
        <div className="sidebar"></div>
      </div>
    </div>
  );
}

export default Landingpage;
