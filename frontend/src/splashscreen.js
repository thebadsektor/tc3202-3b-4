import './App.css';

function App() {
  return (
    <div class="splash-screen">
      <div class="title-intro">
        <h1>BotaniSnap-AI</h1>
        <p class="tagline">Discover plants, Detect Diseases, Nurture Nature.<br>
        </br>Developed by The Hood Reaserch Department,<br>
          </br>powered by LLM (Lots of Lil Money)<br>
          </br>and Nigga Chain AI, with Broke Nigga Detection Tool
        </p>
        <button class="get-started-button" onClick="{handleGetStarted}">
    Get Started
  </button>
      </div>
      <div class="main-intro">
        <p>BotaniSnap-AI is a web application designed to identify plant species<br></br>and detect potential diseases using AI-powered image recognition.<br>
    </br> By simply taking or uploading a photo, you can now quickly determine<br></br>the name of a plant and detect any visible signs of disease.</p>
      </div>
</div>
  );
}

export default App;
