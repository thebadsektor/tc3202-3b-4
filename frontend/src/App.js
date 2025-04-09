// src/App.js
import React, { useState, useEffect } from 'react';
import { onAuthStateChanged, signOut } from "firebase/auth";
import Login from './Login';
import Create from './Create';
import Landingpage from './Landingpage'; // 1. Import Landingpage
import { auth } from "./firebaseConfig";
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [isRegistering, setIsRegistering] = useState(false);
  const [loading, setLoading] = useState(true);

  // Listener for Authentication State Changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
      console.log("Auth State Changed:", currentUser);
    });
    return () => unsubscribe();
  }, []);

  // Handlers
  const handleLogout = async () => {
    try {
      await signOut(auth);
      console.log("User signed out");
      // User state will be set to null by onAuthStateChanged listener
    } catch (error) {
      console.error("Error signing out:", error);
      alert("Failed to sign out.");
    }
  };

  // These success handlers might not be strictly needed for navigation anymore,
  // as onAuthStateChanged handles the user state change which triggers the re-render.
  // But they are useful for logging or other immediate post-login/register actions.
  const handleLoginSuccess = (loggedInUser) => {
    console.log("Login successful in App.js");
  }

  const handleRegisterSuccess = (registeredUser) => {
      console.log("Registration successful in App.js");
      setIsRegistering(false); // Switch back to the login view after registration
  }

  // Render Logic
  if (loading) {
    return <div>Loading...</div>;
  }

  // If user is logged in, show the Landingpage
  if (user) {
    return (
      <div className="App">
        {/* 2. Render Landingpage and pass user/logout handler */}
        <Landingpage user={user} onLogout={handleLogout} />
      </div>
    );
  }

  // If user is not logged in, show Login or Register form
  return (
    <div className="App auth-container">
      {isRegistering ? (
        <Create
          onBackToLogin={() => setIsRegistering(false)}
          onRegisterSuccess={handleRegisterSuccess}
        />
      ) : (
        <Login
          onSwitchToRegister={() => setIsRegistering(true)}
          onLoginSuccess={handleLoginSuccess}
        />
      )}
    </div>
  );
}

export default App;