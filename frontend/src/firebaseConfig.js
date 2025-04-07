// src/firebaseConfig.js
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth"; // Import getAuth

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCmacT3XL8IRyDuoDquM7-hK6XFsEB__3o", // Keep your API keys secure!
  authDomain: "botanisnap.firebaseapp.com",
  databaseURL: "https://botanisnap-default-rtdb.firebaseio.com",
  projectId: "botanisnap",
  storageBucket: "botanisnap.appspot.com",
  messagingSenderId: "939576632048",
  appId: "1:939576632048:web:74fef4aaa34d2dabba1835",
  measurementId: "G-4FBT834QVK"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// Analytics is optional, include if you use it
const analytics = getAnalytics(app);
const auth = getAuth(app); // Initialize Firebase Auth and get the instance

// Export the auth instance to use in other components
export { auth, app, analytics };