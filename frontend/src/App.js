// src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { onAuthStateChanged, signOut } from "firebase/auth";
import Login from './Login';
import Create from './Create';
import SplashScreen from './SplashScreen'; // Add this import
import Landingpage from './Landingpage';
import AboutPage from './AboutPage'; // ✅ Import AboutPage
import { auth } from "./firebaseConfig";
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [isRegistering, setIsRegistering] = useState(false);
  const [showSplash, setShowSplash] = useState(true);
  const [loading, setLoading] = useState(true);


  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);
  
  const handleLogout = async () => {
    try {
      await signOut(auth);
    } catch (error) {
      alert("Failed to sign out.");
    }
  };

  const handleLoginSuccess = () => {
    console.log("Login successful");
  };

  const handleRegisterSuccess = () => {
    setIsRegistering(false);
  };

  if (showSplash) return <SplashScreen onStart={() => setShowSplash(false)} />;
  if (loading) return <div>Loading...</div>;
  

  return (
    <Router>
      <div className="App">
      <Routes>
        {/* Protected Route */}
        {user && (
          <>
            <Route path="/" element={<Landingpage user={user} onLogout={handleLogout} />} />
            <Route path="/about" element={<AboutPage user={user} onLogout={handleLogout} />} />
            <Route path="*" element={<Navigate to="/" />} />
          </>
        )}
        {/* Public Routes */}
        {!user && (
          <>
            <Route path="/" element={
              isRegistering ? (
                <Create
                  onBackToLogin={() => setIsRegistering(false)}
                  onRegisterSuccess={handleRegisterSuccess}
                />
              ) : (
                <Login
                  onSwitchToRegister={() => setIsRegistering(true)}
                  onLoginSuccess={handleLoginSuccess}
                />
              )
            } />
            <Route path="*" element={<Navigate to="/" />} />
          </>
        )}
      </Routes>

      </div>
    </Router>
  );
}

export default App;
