/**
 * EVA Frontend Configuration
 *
 * Firebase and backend configuration.
 * Replace these values with your actual Firebase project config.
 *
 * =========================================================================
 * TEMPORARY DEVELOPMENT NOTE
 * =========================================================================
 * During development, this frontend is served directly by the FastAPI
 * backend via StaticFiles. In production, this frontend will be deployed
 * separately via Firebase Hosting, and the baseUrl below should point to
 * the Cloud Run backend URL.
 * =========================================================================
 */

// Firebase configuration - REPLACE with your Firebase project settings
const firebaseConfig = {
    apiKey: "YOUR_FIREBASE_API_KEY",
    authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT_ID.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID"
};

// Backend API configuration
const API_CONFIG = {
    // Base URL for the backend API - update for production
    baseUrl: window.location.hostname === 'localhost'
        ? 'http://localhost:8080'
        : window.location.origin,
    apiVersion: '/api/v1',
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
