// src/SplashScreen.js
import React from 'react';
import './splashscreen.css';

function SplashScreen({ onGetStarted }) {
  return (
    <div className="splash-screen">
      <div className="title-intro">
        <h1>BotaniSnap-AI</h1>
        <p className="tagline">
          Discover plants. Detect diseases. Nurture nature.<br />
          Built by the Hood Research Department,<br />
          powered by creativity, community, and cutting-edge AI.<br />
          Featuring our Smart Health Detection Engine.
        </p>
        <button className="get-started-button" onClick={onGetStarted}>
          Get Started
        </button>
      </div>
      <div className="main-intro">
        <p>
          BotaniSnap-AI is a web application designed to identify plant<br />
          species and detect potential diseases using AI-powered image<br />
          recognition. By simply taking or uploading a photo, you can<br />
          quickly determine the name of a plant and spot signs of illness.
        </p>
      </div>
    </div>
  );
}

export default SplashScreen;
