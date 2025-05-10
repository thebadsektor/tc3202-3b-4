// src/Landingpage.js
import React, { useState, useRef } from "react";
import axios from "axios";
import "./Landingpage.css";
import Webcam from 'react-webcam';
import  { marked } from 'marked'
import { useNavigate } from 'react-router-dom';

function Landingpage({ user, onLogout }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [facingMode] = useState("environment"); // Default to rear camera
  const [mode, setMode] = useState("identify"); // "identify" or "disease"
  const [diseaseResults, setDiseaseResults] = useState(null);
  const navigate = useNavigate();
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  const getVideoConstraints = () => {
    return {
      facingMode: { ideal: facingMode }, // 'user' or 'environment'
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    };
  };
  
  const resizeImage = async (file, maxWidth = 1280, maxHeight = 720) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.src = URL.createObjectURL(file);

      img.onload = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
        const width = img.width * ratio;
        const height = img.height * ratio;

        canvas.width = width;
        canvas.height = height;

        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob(blob => {
          const resizedFile = new File([blob], file.name, { type: "image/jpeg" });
          resolve(resizedFile);
        }, "image/jpeg", 0.9); // quality 90%
      };

      img.onerror = reject;
    });
  };

  const startCamera = async () => {
    try {
      const constraints = getVideoConstraints();
      await navigator.mediaDevices.getUserMedia({ video: constraints });
      setIsCameraActive(true);
      setError(null);
    } catch (err) {
      console.error("Camera access error:", err);
      setError("Camera access denied. Please allow camera permissions.");
    }
  };
  
  const stopCamera = () => {
    setIsCameraActive(false);
  };

  const captureImage = () => {
    if (!webcamRef.current) {
      setError("Camera not properly initialized");
      return;
    }

    try {
      // Capture image from webcam
      const imageSrc = webcamRef.current.getScreenshot();
      
      if (!imageSrc) {
        setError("Failed to capture image. Please try again.");
        return;
      }
      
      // Convert data URL to blob/file
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const capturedImage = new File([blob], "captured_image.jpg", { type: "image/jpeg" });
          setSelectedFile(capturedImage);
          setPreview(imageSrc);
          stopCamera();
          setPrediction(null); // Clear any previous predictions
          setDiseaseResults(null); // Clear any previous disease results
        })
        .catch(err => {
          console.error("Error processing capture:", err);
          setError("Failed to process captured image: " + err.message);
        });
    } catch (err) {
      console.error("Error capturing image:", err);
      setError("Failed to capture image: " + err.message);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setDiseaseResults(null);
      setError(null);
      stopCamera();
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select or capture an image first.");
      return;
    }
    const formData = new FormData();

    // Resize the image before uploading
    const resizedFile = await resizeImage(selectedFile);  // <--- Your helper function
    formData.append("file", resizedFile);
    

    setLoading(true);
    setError(null);

    try {
      if (mode === "identify") {
        // Plant identification API call
        const response = await axios.post("http://localhost:5000/predict", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setPrediction(response.data);
        setDiseaseResults(null);
      } else {
        // Disease detection API call
        const response = await axios.post("http://localhost:5000/predict_disease_hf", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setDiseaseResults(response.data);
        setPrediction(null);
      }
    } catch (err) {
      console.error("Upload error:", err);
      setError(`Failed to ${mode === "identify" ? "identify plant" : "detect disease"}. Please check your connection and try again.`);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = () => {
    document.getElementById('fileInput').click();
  };

  const resetCapture = () => {
    setPreview(null);
    setSelectedFile(null);
    setPrediction(null);
    setDiseaseResults(null);
    setError(null);
  };

  // Toggle between plant identification and disease detection modes
  const toggleMode = () => {
    setMode(mode === "identify" ? "disease" : "identify");
    setPrediction(null);
    setDiseaseResults(null);
  };

  // Format the disease information for display
  const formatDiseaseInfo = (info) => {
    if (!info || !info.info) return null;
    
    return info.info.split('\n').map((paragraph, index) => (
      <p key={index} className="plant-info-text">{paragraph}</p>
    ));
  };

  // Render disease detection results
  const renderDiseaseResults = () => {
    if (!diseaseResults) return null;
    
    const { prediction, confidence_level, disease_info } = diseaseResults;
    const confidencePercentage = (confidence_level * 100).toFixed(2);
    
    return (
      <div className="prediction-results">
        <h3>Disease Detection Results:</h3>
        <p>Diagnosis: <strong>{prediction.replace('_', ' ')}</strong></p>
        <p>Confidence: {confidencePercentage}%</p>
        
        {prediction === "Healthy" ? (
          <div className="healthy-message">
            <p>Good news! Your plant appears to be healthy.</p>
            <p>Continue with proper care and regular monitoring.</p>
          </div>
        ) : (
          <div className="disease-info-container">
            <h3>Disease Information:</h3>
            {disease_info && disease_info.status === "success" ? (
              <div className="disease-info-content">
                {formatDiseaseInfo(disease_info)}
              </div>
            ) : (
              <p>No detailed information available for this disease.</p>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="landing-container">
      <header className="landing-header">
        <div className="logo-header">
        <img
          src={`${process.env.PUBLIC_URL}/logo.png`}
          alt="BotaniSnap-AI Logo"
          className="logo-image"
        />
        <h2 className="BotaniText">BotaniSnap-AI</h2>
      </div>

        <div className="user-menu">
  <div className="menu-trigger" onClick={() => setMenuOpen(!menuOpen)}>
  <div
  className="profile-circle"
  style={{
    backgroundImage: `url(${process.env.PUBLIC_URL + '/profile.jpg'})`,
    width: '50px',
    height: '50px',
    borderRadius: '50%',
    backgroundSize: 'cover',
    backgroundPosition: 'center'
  }}
></div>

  </div>
  {menuOpen && (
    <div className="dropdown-menu">
      <div className="profile-info">
        <div
      className="profile-circle-large"
      style={{
        backgroundImage: `url(${process.env.PUBLIC_URL + '/profile.jpg'})`,
        width: '40px',
        height: '40px',
        borderRadius: '50%',
        backgroundSize: 'cover',
        backgroundPosition: 'center'
      }}
        ></div>
        <p>{user ? user.email : "User"}</p>
      </div>
      <button className="dropdown-item" onClick={() => navigate('/about')}>About</button>
      <button className="dropdown-item logout" onClick={onLogout}>Log Out</button>
    </div>
  )}
</div>

      </header>
      <div className="landing-main">
        <div className="main-box">
          {/* Mode indicator */}
          <div className="mode-indicator">
            <h3>Mode: {mode === "identify" ? "Plant Identification" : "Disease Detection"}</h3>
            <button className="mode-toggle-button" onClick={toggleMode}>
              Switch to {mode === "identify" ? "Disease Detection" : "Plant Identification"}
            </button>
          </div>

          {/* Camera Preview Section using react-webcam */}
          {isCameraActive && (
            <div className="camera-container">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                screenshotQuality={1}
                videoConstraints={getVideoConstraints()}
                className="camera-preview"
                mirrored={facingMode === "user"}
                playsInline // ✅ This is crucial for iOS
                onUserMediaError={(err) => {
                  console.error("Webcam error:", err);
                  setError("Camera access error: " + (err.message || "Could not access camera"));
                  stopCamera();
                }}
              />
              <div className="camera-controls">
                <button 
                  className="action-button" 
                  onClick={captureImage}
                >
                  Capture
                </button>
              
                <button 
                  className="action-button" 
                  onClick={stopCamera}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Initial State - No Camera or Preview */}
          {!isCameraActive && !preview && (
            <div className="camera-controls">
              <button 
                className="action-button" 
                onClick={startCamera}
              >
                Open Camera
              </button>
              <input 
                type="file" 
                id="fileInput"
                accept="image/*" 
                onChange={handleFileChange} 
                style={{display: 'none'}} 
              />
              <button 
                className="action-button" 
                onClick={handleFileUpload}
              >
                Upload Photo
              </button>
            </div>
          )}

          {/* Preview Section */}
          {preview && (
            <div className="preview-container">
              <img 
                src={preview} 
                alt="Preview" 
                className="preview-image" 
              />
              <div className="camera-controls">
                <button 
                  onClick={handleUpload} 
                  disabled={loading} 
                  className="action-button"
                >
                  {loading ? "Processing..." : mode === "identify" ? "Identify Plant" : "Detect Disease"}
                </button>
                <button 
                  className="action-button" 
                  onClick={resetCapture}
                >
                  New Photo
                </button>
              </div>
            </div>
          )}

          <canvas 
            ref={canvasRef} 
            style={{display: 'none'}} 
          />
        </div>
        
        <div className="side-box">
        {error && (
          <div className="error-message">
            <p className="error">{error}</p>
            {error.includes("denied") && (
              <p className="small-text">
                Tip: Check if your browser has permission to access the camera. On iOS, go to Settings &gt; Safari &gt; Camera and set to "Allow".
              </p>
            )}
            <button 
              className="action-button small" 
              onClick={startCamera}
            >
              Retry Camera
            </button>
          </div>
        )}
          {loading && (
  <div className="loading-indicator">
    <div className="loader-animation"></div> {/* This is your spinner or animation */}
    <div className="loading-text">
      {mode === "identify" ? "Analyzing your plant image..." : "Checking for plant diseases..."}
    </div>
  </div>
)}


          {/* Plant Identification Results */}
          {!loading && prediction && (
   <div className="prediction-results">
   {prediction.accuracy < 60 ? (
     <div className="low-confidence-message">
       <h3>Sorry. I couldn't find this plant. Please Take or Upload another photo</h3>
       <button 
         className="action-button" 
         onClick={resetCapture}
       >
         Try Again
       </button>
     </div>
   ) : (
     <>
       <h2 className="text-xl font-bold">Identified Plant: {prediction.predicted_plant}</h2>
       
       {prediction.plant_info && prediction.plant_info.status === "success" && (
         <div className="plant-info-container mt-4 p-4 bg-white rounded shadow">
           <h3 className="text-green-600 font-semibold text-lg mb-2">Plant Information:</h3>
   
           <div className="plant-info-content prose max-w-none"
             dangerouslySetInnerHTML={{
               __html: marked.parse(prediction.plant_info.info),
             }}
           />
         </div>
       )}
     </>
   )}
 </div>
)}
          {/* Disease Detection Results */}
          {diseaseResults && renderDiseaseResults()}
        </div>
      </div>
    </div>
  );
}

export default Landingpage;