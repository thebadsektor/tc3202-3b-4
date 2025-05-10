// src/Login.js
import React, { useState } from "react";
import { signInWithEmailAndPassword, signInWithPopup, GoogleAuthProvider, FacebookAuthProvider } from "firebase/auth";
import { auth } from "./firebaseConfig"; // Assuming firebaseConfig.js is in src/
import "./Login.css";                  // Make sure this CSS file has the styles from your design

function Login({ onSwitchToRegister, onLoginSuccess }) {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [error, setError] = useState(""); // State for error messages

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setError(""); // Clear error on change
  };

  // Firebase Sign In Logic
  const handleEmailPasswordLogin = async (e) => {
    e.preventDefault(); // Prevent default form submission
    setError(""); // Clear previous errors

    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        formData.email,
        formData.password
      );
      const user = userCredential.user;
      console.log("Logged in successfully via Email/Password:", user);

      if (onLoginSuccess) {
        onLoginSuccess(user);
      } else {
         alert('Login successful!');
      }

    } catch (error) {
      console.error("Error logging in:", error);
       if (error.code === 'auth/user-not-found' || error.code === 'auth/wrong-password' || error.code === 'auth/invalid-credential') {
         setError('Invalid email or password.');
       } else if (error.code === 'auth/invalid-email') {
         setError('Please enter a valid email address.');
       } else {
         setError('Failed to log in. Please try again.');
       }
    }
  };

  // Firebase Sign In Logic for Google Login
  const handleGoogleLogin = async () => {
    setError(""); // Clear previous errors
    const provider = new GoogleAuthProvider();

    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;
      console.log("Logged in successfully via Google:", user);
      if (onLoginSuccess) {
        onLoginSuccess(user);
      }
    } catch (error) {
      console.error("Error logging in with Google:", error);
      setError('Failed to log in with Google. Please try again.');
    }
  };

  // Firebase Sign In Logic for Facebook Login
  const handleFacebookLogin = async () => {
    setError(""); // Clear previous errors
    const provider = new FacebookAuthProvider();

    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;
      console.log("Logged in successfully via Facebook:", user);
      if (onLoginSuccess) {
        onLoginSuccess(user);
      }
    } catch (error) {
      console.error("Error logging in with Facebook:", error);
      setError('Failed to log in with Facebook. Please try again.');
    }
  };

  return (
    <div className="login-container">
      <h2 className="login-text">BotaniSnap</h2>

      {/* Display Errors */}
      {error && <p className="error-message" style={{color: 'red', textAlign: 'center', marginBottom: '10px'}}>{error}</p>}

      {/* Email/Password Form */}
      <form className="login-form" onSubmit={handleEmailPasswordLogin}>
        <input
          type="email" // Use email type
          name="email" // Name matches state
          placeholder="Email" // Placeholder is Email
          className="input-field"
          value={formData.email}
          onChange={handleChange}
          required
        />
        <input
          type="password"
          name="password" // Name matches state
          placeholder="Password"
          className="input-field"
          value={formData.password}
          onChange={handleChange}
          required
        />
        <button
          type="submit" // Submit button for the form
          className="login-button"
        >
          Login
        </button>
      </form>

      {/* OR Divider */}
      <div className="or-divider">
        <span>OR</span>
      </div>

      {/* Social Login Buttons */}
      <button type="button" className="fb-button" onClick={handleFacebookLogin}>
        Sign in with Facebook
      </button>
      <button type="button" className="google-button" onClick={handleGoogleLogin}>
        Sign in with Google
      </button>

      {/* Link to Create Account */}
      <p className="create-account">
        Don't have an account?{' '}
        <span
          className="signup-link"
          onClick={onSwitchToRegister} // Use the prop to switch view
          style={{cursor: 'pointer', color: 'blue', textDecoration: 'underline'}} // Added basic styling for link
        >
          Create one
        </span>
      </p>
    </div>
  );
}

export default Login;
