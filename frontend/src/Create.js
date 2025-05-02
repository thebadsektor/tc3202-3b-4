import React, { useState } from "react";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { getDatabase, ref, set } from "firebase/database";
import { auth } from "./firebaseConfig";
import "./Create.css";

function Create({ onBackToLogin, onRegisterSuccess }) {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: ""
  });
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setError(""); // Clear general error
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    // --- Validation ---
    const emptyFields = Object.entries(formData).filter(([_, val]) => val.trim() === "");
    if (emptyFields.length > 0) {
      setError("Please fill out all fields.");
      return;
    }
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (formData.password.length < 6) {
      setError("Password should be at least 6 characters.");
      return;
    }

    // --- Create User in Firebase ---
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        formData.email,
        formData.password
      );
      const user = userCredential.user;

      // --- Save to Realtime Database ---
      const db = getDatabase();
      await set(ref(db, "users/" + user.uid), {
        username: formData.username,
        email: formData.email,
        uid: user.uid,
        createdAt: new Date().toISOString()
      });

      if (onRegisterSuccess) {
        onRegisterSuccess(user);
      } else {
        alert("Registration successful!");
        if (onBackToLogin) onBackToLogin();
      }
    } catch (error) {
      console.error("Registration error:", error);
      if (error.code === "auth/email-already-in-use") {
        setError("This email address is already registered.");
      } else if (error.code === "auth/invalid-email") {
        setError("Please enter a valid email.");
      } else if (error.code === "auth/weak-password") {
        setError("Password is too weak.");
      } else {
        setError("Failed to create account. Try again.");
      }
    }
  };

  return (
    <div className="create-container">
      <h2 className="create-text">Create Account</h2>
      {error && <p className="error-message" style={{ color: "red" }}>{error}</p>}
      <form className="create-form" onSubmit={handleSubmit}>
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
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          className="input-field"
          value={formData.password}
          onChange={handleChange}
        />
        <input
          type="password"
          name="confirmPassword"
          placeholder="Confirm Password"
          className="input-field"
          value={formData.confirmPassword}
          onChange={handleChange}
        />
        <button type="submit" className="create-button">Create Account</button>
      </form>
      <div className="login-link-container">
        <p>
          Already have an account?{" "}
          <span
            className="login-link"
            onClick={onBackToLogin}
            style={{ cursor: "pointer", color: "blue", textDecoration: "underline" }}
          >
            Login
          </span>
        </p>
      </div>
    </div>
  );
}

export default Create;
