import React from 'react';
import './App.css';

function SplashScreen({ onGetStarted }) {
  return (
    <div className="splash-screen">
      <div className="title-intro">
        <h1>BotaniSnap-AI</h1>
        <p className="tagline">
          Discover plants, Detect Diseases, Nurture Nature.<br />
          Developed by The Hood Research Department,<br />
          powered by LLM (Lots of Lil Money)<br />
          and [Custom AI Name Placeholder], with Advanced Detection Tools
        </p>
        <button className="get-started-button" onClick={onGetStarted}>
          Get Started
        </button>
      </div>
      <div className="main-intro">
        <p>
          BotaniSnap-AI is a web application designed to identify plant species<br />
          and detect potential diseases using AI-powered image recognition.<br />
          By simply taking or uploading a photo, you can now quickly determine<br />
          the name of a plant and detect any visible signs of disease.
        </p>
      </div>
    </div>
  );
}

export default SplashScreen;
