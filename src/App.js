import React, { useState } from "react";
import Header from "./Header";
import Login from "./Login";
import Landingpage from "./Landingpage";

function App() {
  const [loggedIn, setLoggedIn] = useState(false);

  // 🔹 Logout function (resets loggedIn state)
  const handleLogout = () => {
    setLoggedIn(false);
  };

  return (
    <div className="App">
      {loggedIn ? (
        <Landingpage onLogout={handleLogout} />  // 🔹 Pass logout function
      ) : (
        <Login onLogin={() => setLoggedIn(true)} />
      )}
    </div>
  );
}

export default App;