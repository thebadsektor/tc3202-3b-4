// src/AboutPage.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './AboutPage.css';

function AboutPage({ user, onLogout }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const navigate = useNavigate();

  return (
    <>
      {/* Full-width, sticky header */}
      <header className="landing-header">
        <h2>BotaniSnap-AI</h2>
        <div className="user-menu">
          <div className="menu-trigger" onClick={() => setMenuOpen(!menuOpen)}>
            <div className="profile-circle"></div>
          </div>
          {menuOpen && (
            <div className="dropdown-menu">
              <div className="profile-info">
                <div className="profile-circle-large"></div>
                <p>{user ? user.email : "User"}</p>
              </div>
              <button className="dropdown-item" onClick={() => navigate('/')}>Back to App</button>
              <button className="dropdown-item logout" onClick={onLogout}>Log Out</button>
            </div>
          )}
        </div>
      </header>

      {/* Container limited in width */}
      <div className="about-container">
        <main className="about-content">
          <section>
            <h2>Our Mission</h2>
            <p>
              BotaniSnap-AI aims to make plant identification and plant health assessment easy and accessible for everyone...
            </p>
          </section>

          <section>
            <h2>How It Works</h2>
            <p>
              Simply upload a photo or use your device's camera to capture an image...
            </p>
            <ul>
              <li><strong>Plant Identification:</strong> Identifies the plant species with details about it.</li>
              <li><strong>Disease Detection:</strong> Detects common plant diseases and provides information about them.</li>
            </ul>
          </section>

          <section>
            <h2>Technology Used</h2>
            <p>This application is built using modern web technologies:</p>
            <ul>
              <li><strong>Frontend:</strong> React.js, CSS, react-webcam</li>
              <li><strong>Backend:</strong> Flask/Python</li>
              <li><strong>AI Models:</strong> BotaniSnap Model(Our Trained Model),</li>
              <li>Plant-Disease-Detection-Project from Hugging Face,</li>
              <li>Gemini flash 2.0 for Plant Description and etc.</li>
            </ul>
          </section>

          <section>
            <h2>Developed By Group 4</h2>
            <p>Cantal Marc Airon T.</p>
            <p>Salabsab Sean Richard</p>
            <p>Mangalindan Giro</p>
            <p>Engr. Canaling John Jasper Jr.</p>
            <p>Contact: Sulasok.tv</p>
          </section>
        </main>

        <footer className="about-footer">
          <p>© {new Date().getFullYear()} BotaniSnap-AI. All rights reserved.</p>
        </footer>
      </div>
    </>
  );
}

export default AboutPage;
