// src/AboutPage.js
import React from 'react';
import { Link } from 'react-router-dom'; // Import Link for navigation
import './AboutPage.css'; // We'll create this CSS file next

function AboutPage() {
  return (
    <div className="about-container">
      <header className="about-header">
        {/* You might want a consistent header/nav bar across pages later */}
        <h1>About BotaniSnap-AI</h1>
        {/* Provide a link back to the main landing page */}
        <Link to="/" className="back-link">← Back to App</Link>
      </header>

      <main className="about-content">
        <section>
          <h2>Our Mission</h2>
          <p>
            BotaniSnap-AI aims to make plant identification and plant health assessment easy and accessible for everyone. Whether you're a seasoned gardener, a curious nature enthusiast, or just trying to figure out what's growing in your backyard, our app provides quick and informative results powered by AI.
          </p>
        </section>

        <section>
          <h2>How It Works</h2>
          <p>
            Simply upload a photo or use your device's camera to capture an image of a plant or a potentially diseased leaf. Our AI models will analyze the image and provide:
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
            <li><strong>Backend (example):</strong> Flask/Python (handling the AI models)</li>
            <li><strong>AI Models:</strong> Deep Learning models trained for image classification (plant species and diseases).</li>
            {/* Add specific model libraries if you want, e.g., TensorFlow, PyTorch */}
          </ul>
        </section>

        <section>
          <h2>Developed By</h2>
          <p>
            [Your Name/Your Team Name]
            {/* Optional: Add links to portfolio, GitHub, etc. */}
          </p>
          <p>
            Contact: [Your Contact Email or Link]
          </p>
        </section>
      </main>

      <footer className="about-footer">
        <p>© {new Date().getFullYear()} BotaniSnap-AI. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default AboutPage;