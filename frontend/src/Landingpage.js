import React, { useState, useRef } from "react";
import axios from "axios";
import "./Landingpage.css";

function Landingpage({ onLogout }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (err) {
      setError("Camera access denied.");
    }
  };

  const captureImage = () => {
    if (canvasRef.current && videoRef.current) {
      const context = canvasRef.current.getContext('2d');
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      
      context.drawImage(videoRef.current, 0, 0, 
        canvasRef.current.width, 
        canvasRef.current.height);
      
      // Convert canvas to blob and create file
      canvasRef.current.toBlob((blob) => {
        const file = new File([blob], "captured_image.jpg", { type: "image/jpeg" });
        setSelectedFile(file);
        setPreview(canvasRef.current.toDataURL());
        setIsCameraActive(false);
        
        // Stop video tracks
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        videoRef.current.srcObject = null;
      });
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
      
      // Stop camera if active
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        videoRef.current.srcObject = null;
        setIsCameraActive(false);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
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
      setError("Failed to get prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = () => {
    document.getElementById('fileInput').click();
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    }
  };

  return (
    <div className="landing-container">
      <header className="landing-header">
        <h2>BotaniSnap-AI</h2>
        <div className="user-menu">
          <span>User</span>
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
          {/* Camera Preview Section */}
          {isCameraActive && (
            <div className="camera-container">
              <video 
                ref={videoRef} 
                className="camera-preview" 
                autoPlay 
                playsInline 
                muted
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
                  {loading ? "Processing..." : "Upload & Predict"}
                </button>
                <button 
                  className="action-button" 
                  onClick={() => {
                    setPreview(null);
                    setSelectedFile(null);
                  }}
                >
                  Clear
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
          {error && <p className="error">{error}</p>}

          {prediction && (
            <div className="prediction-results">
              <h2>Prediction: {prediction.predicted_plant}</h2>
              <p>Confidence: {prediction.accuracy}%</p>
              <h3>Top Predictions:</h3>
              <ul>
                {prediction.top_predictions.map((item, index) => (
                  <li key={index}>
                    {item.plant_name}: {item.accuracy}%
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Landingpage;