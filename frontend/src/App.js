import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file)); // Show preview
      setPrediction(null);
      setError(null);
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
      setPrediction(response.data); // Store entire response object
    } catch (err) {
      setError("Failed to get prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <div className="buttonheader">
          <div className="logo">BotaniSnap</div>
          <ul><li>Home</li></ul>
          <ul><li>About</li></ul>
          <ul><li>Gallery</li></ul>
          <ul><li>Contact</li></ul>
        </div>
      </div>

      <div className="card">
        <h1 className="text-2xl font-bold mb-4">Upload your Plant Image</h1>
        <input type="file" accept="image/*" onChange={handleFileChange} className="mb-4" />
        
        {/* {preview && <img src={preview} alt="Preview" className="preview-image" />}  */}

        {preview && <img src={preview} alt="Preview" className="preview-image" style={{ width: "300px", height: "300px", objectFit: "cover" }} />}

        <button onClick={handleUpload} disabled={loading} className="button">
          {loading ? "Processing..." : "Upload & Predict"}
        </button>

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
  );
};

export default App;
