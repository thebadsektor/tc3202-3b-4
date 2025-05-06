import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { onAuthStateChanged, signOut } from "firebase/auth";
import Login from './Login';
import Create from './Create';
import Landingpage from './Landingpage';
import AboutPage from './AboutPage';
import SplashScreen from './splashscreen'; // ✅ Import SplashScreen
import { auth } from "./firebaseConfig";
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [isRegistering, setIsRegistering] = useState(false);
  const [loading, setLoading] = useState(true);
  const [showSplash, setShowSplash] = useState(true); // ✅ Splash screen state

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

  const handleGetStarted = () => {
    setShowSplash(false); // ✅ Hide splash screen
  };

  if (loading) return <div>Loading...</div>;

  if (showSplash) return <SplashScreen onGetStarted={handleGetStarted} />; // ✅ Show splash

  return (
    <Router>
      <div className="App">
        <Routes>
          {user && (
            <>
              <Route path="/" element={<Landingpage user={user} onLogout={handleLogout} />} />
              <Route path="/about" element={<AboutPage user={user} onLogout={handleLogout} />} />
              <Route path="*" element={<Navigate to="/" />} />
            </>
          )}
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
