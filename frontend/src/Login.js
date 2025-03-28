import React from "react";
import "./Login.css";

function Login({ onLogin }) {
  const handleLogin = () => {
    onLogin(true);
  };

  return (
    <div className="login-container">
      <h2 className="login-text">BotaniSnap</h2>
      <div className="login-form">
        <input
          type="text"
          placeholder="Username"
          className="input-field"
        />
        <input
          type="password"
          placeholder="Password"
          className="input-field"
        />
        <button
          className="login-button"
          onClick={handleLogin}
        >
          Login
        </button>
      </div>
      <div className="or-divider">
        <span>OR</span>
      </div>
      <button className="fb-button">
        Sign in with Facebook
      </button>
      <button className="google-button">
        Sign in with Google
      </button>
      <p className="create-account">
        Don't have an account? <span className="signup-link">Create one</span>
      </p>
    </div>
  );
}

export default Login;