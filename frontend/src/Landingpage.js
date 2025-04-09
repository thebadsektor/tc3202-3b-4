// src/Landingpage.js
import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./Landingpage.css";
import Webcam from 'react-webcam';

function Landingpage({ user, onLogout }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState("environment"); // Default to rear camera
  const [availableCameras, setAvailableCameras] = useState([]);
  const [currentCameraIndex, setCurrentCameraIndex] = useState(0);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Enumerate available cameras
  const getCameras = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      setAvailableCameras(videoDevices);
      console.log("Available cameras:", videoDevices);
    } catch (err) {
      console.error("Error enumerating devices:", err);
      setError("Unable to detect available cameras: " + err.message);
    }
  };

  // Initial setup
  useEffect(() => {
    // Check for camera support and enumerate devices
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      getCameras();
    } else {
      console.error("MediaDevices API not supported");
      setError("Your browser doesn't support camera access");
    }
  }, []);

  const startCamera = () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: { facingMode } })
        .then(stream => {
          setIsCameraActive(true);
          setError(null);
          // You can optionally display the camera stream in a video element or use react-webcam as you are doing
        })
        .catch(err => {
          console.error("Camera access error:", err);
          setError("Camera access denied. Please allow camera permissions.");
        });
    } else {
      setError("Camera access is not supported by your browser.");
    }
  };
  

  const stopCamera = () => {
    setIsCameraActive(false);
  };

  const switchCamera = () => {
    if (availableCameras.length > 1) {
      // If we have multiple cameras, cycle through them
      const nextCameraIndex = (currentCameraIndex + 1) % availableCameras.length;
      setCurrentCameraIndex(nextCameraIndex);
      console.log("Switching to camera index:", nextCameraIndex);
    } else {
      // If we don't have specific camera info, toggle between front and back
      const newFacingMode = facingMode === "user" ? "environment" : "user";
      setFacingMode(newFacingMode);
      console.log("Switching facing mode to:", newFacingMode);
    }
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
    formData.append("file", selectedFile);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post("http://localhost:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(response.data);
    } catch (err) {
      console.error("Upload error:", err);
      setError("Failed to get prediction. Please check your connection and try again.");
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
    setError(null);
  };

  // Get video constraints based on selected camera
  const getVideoConstraints = () => {
    if (availableCameras.length > 0 && availableCameras[currentCameraIndex]?.deviceId) {
      return {
        deviceId: availableCameras[currentCameraIndex].deviceId,
        facingMode: undefined,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      };
    } else {
      return {
        facingMode: facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      };
    }
  };

  return (
    <div className="landing-container">
      <header className="landing-header">
        <h2>BotaniSnap-AI</h2>
        <div className="user-menu">
          <span>{user ? user.email : 'User'}</span>
          <div className="user-icon"></div>
          <div className="menu-icon" onClick={() => setMenuOpen(!menuOpen)}>
            &#9776;
          </div>
          {menuOpen && (
            <div className="dropdown-menu">
              <ul>
                <li>Profile</li>
                <li>Menu</li>
                <li>Gallery</li>
                <li className="logout" onClick={onLogout}>Log Out</li>
              </ul>
            </div>
          )}
        </div>
      </header>

      <div className="landing-main">
        <div className="main-box">
          {/* Camera Preview Section using react-webcam */}
          {isCameraActive && (
            <div className="camera-container">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={getVideoConstraints()}
                className="camera-preview"
                mirrored={facingMode === "user"}
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
                  onClick={switchCamera}
                >
                  Switch Camera ({availableCameras.length > 1 ? 
                    `Camera ${currentCameraIndex + 1}/${availableCameras.length}` : 
                    facingMode === "user" ? "Front" : "Back"})
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
                  {loading ? "Processing..." : "Identify Plant"}
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
        
        <div className="sidebar">
          {error && (
            <div className="error-message">
              <p className="error">{error}</p>
              {isCameraActive && (
                <button 
                  className="action-button small" 
                  onClick={startCamera}
                >
                  Retry Camera
                </button>
              )}
            </div>
          )}
          
          {loading && (
            <div className="loading-indicator">
              Analyzing your plant image...
            </div>
          )}

          {prediction && (
            <div className="prediction-results">
              <h2>Identified Plant: {prediction.predicted_plant}</h2>
              
              {/* Display plant info from Gemini */}
              {prediction.plant_info && prediction.plant_info.status === "success" && (
                <div className="plant-info-container">
                  <h2>Plant Information:</h2>
                  <div className="plant-info-content">
                    {prediction.plant_info.info.split('\n').map((paragraph, index) => (
                      <p key={index} className="plant-info-text">{paragraph}</p>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Landingpage;