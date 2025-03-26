import React from "react";
import "./Login.css"; // Make sure you have styles for the buttons

function Login({ onLogin }) {
  return (
    <div className="login-container">
      <h2>Login</h2>

      {/* Username & Password Inputs */}
      <input type="text" placeholder="Username" className="input-field" />
      <input type="password" placeholder="Password" className="input-field" />

      {/* Login Button */}
      <button className="login-button" onClick={onLogin}>
        Login
      </button>

      {/* OR Separator */}
      <div className="or-divider">
        <span>OR</span>
      </div>

      {/* Facebook & Google Login Buttons (Dummy) */}
      <button className="fb-button">Sign in with Facebook</button>
      <button className="google-button">Sign in with Google</button>

      {/* Create Account Link (Dummy) */}
      <p className="create-account">
        Don't have an account? <span className="signup-link">Create one</span>
      </p>
    </div>
  );
}

export default Login;