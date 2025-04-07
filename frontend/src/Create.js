// src/Create.js
import React, { useState } from "react";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { auth } from "./firebaseConfig"; // Corrected path: './'
import "./Create.css";                  // Corrected path: './'

function Create({ onBackToLogin, onRegisterSuccess }) {
  const [formData, setFormData] = useState({
    username: "", // Keep username for potential profile data later
    email: "",
    password: "",
    confirmPassword: ""
  });
  const [error, setError] = useState(""); // State to hold error messages

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setError(""); // Clear error on change
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(""); // Clear previous errors

    // --- Validation ---
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return; // Stop submission
    }
    if (formData.password.length < 6) {
       setError("Password should be at least 6 characters long.");
       return;
    }
    // Add more validation as needed (e.g., email format)

    // --- Firebase Account Creation ---
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth, // Use the imported auth instance
        formData.email,
        formData.password
      );
      // Signed in
      const user = userCredential.user;
      console.log("Account created successfully:", user);
      // You can optionally store the username in Firestore or Realtime Database here
      // associated with user.uid

      // Notify parent component or redirect
      if (onRegisterSuccess) {
        onRegisterSuccess(user); // Pass the user object up
      } else {
        alert('Registration successful!'); // Simple feedback if no handler provided
        if(onBackToLogin) onBackToLogin(); // Go back to login view after success only if function provided
      }

    } catch (error) {
      console.error("Error creating account:", error);
      // Provide user-friendly error messages
      if (error.code === 'auth/email-already-in-use') {
        setError('This email address is already registered.');
      } else if (error.code === 'auth/invalid-email') {
        setError('Please enter a valid email address.');
      } else if (error.code === 'auth/weak-password') {
        setError('Password is too weak. Please use a stronger password.');
      } else {
        setError('Failed to create account. Please try again.');
      }
    }
  };

  return (
    <div className="create-container">
      <h2 className="create-text">Create Account</h2>
      {error && <p className="error-message" style={{color: 'red'}}>{error}</p>} {/* Display errors */}
      <form className="create-form" onSubmit={handleSubmit}>
        {/* Username field (keep it if you plan to store username elsewhere) */}
        <input
          type="text"
          name="username"
          placeholder="Username"
          className="input-field"
          value={formData.username}
          onChange={handleChange}
        />
        <input
          type="email"
          name="email"
          placeholder="Email"
          className="input-field"
          value={formData.email}
          onChange={handleChange}
          required
        />
        <input
          type="password"
          name="password"
          placeholder="Password (min. 6 characters)"
          className="input-field"
          value={formData.password}
          onChange={handleChange}
          required
        />
        <input
          type="password"
          name="confirmPassword"
          placeholder="Confirm Password"
          className="input-field"
          value={formData.confirmPassword}
          onChange={handleChange}
          required
        />
        <button type="submit" className="create-button">
          Create Account
        </button>
      </form>
      <div className="login-link-container">
        <p>
          Already have an account?{" "}
          <span className="login-link" onClick={onBackToLogin} style={{cursor: 'pointer', color: 'blue', textDecoration: 'underline'}}>
            Login
          </span>
        </p>
      </div>
    </div>
  );
}

export default Create;