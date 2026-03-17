/**
 * EVA Authentication Module
 *
 * Handles Google sign-in via Firebase Auth and token management.
 */

let currentUser = null;
let authToken = null;

/**
 * Sign in with Google using Firebase Auth.
 */
async function signInWithGoogle() {
    const loginBtn = document.getElementById('login-btn');
    const loginError = document.getElementById('login-error');
    loginError.style.display = 'none';

    try {
        loginBtn.disabled = true;
        loginBtn.textContent = 'Signing in...';

        const provider = new firebase.auth.GoogleAuthProvider();
        const result = await firebase.auth().signInWithPopup(provider);
        const idToken = await result.user.getIdToken();

        // Send Firebase token to backend for verification
        const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.apiVersion}/auth/firebase`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id_token: idToken }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Authentication failed (${response.status})`);
        }

        const data = await response.json();
        authToken = data.access_token;
        currentUser = data.user;

        // Store token for session persistence
        sessionStorage.setItem('eva_token', authToken);
        sessionStorage.setItem('eva_user', JSON.stringify(currentUser));

        showChatPage();
    } catch (error) {
        console.error('Sign-in error:', error);
        loginError.textContent = error.message || 'Sign-in failed. Please try again.';
        loginError.style.display = 'block';
    } finally {
        loginBtn.disabled = false;
        loginBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
                <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z" fill="#4285F4"/>
                <path d="M9.003 18c2.43 0 4.467-.806 5.956-2.18l-2.909-2.26c-.806.54-1.837.86-3.047.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332C2.438 15.983 5.482 18 9.003 18z" fill="#34A853"/>
                <path d="M3.964 10.712c-.18-.54-.282-1.117-.282-1.71 0-.593.102-1.17.282-1.71V4.96H.957C.347 6.175 0 7.55 0 9.002c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
                <path d="M9.003 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.464.891 11.428 0 9.002 0 5.48 0 2.438 2.017.956 4.958L3.964 7.29c.708-2.127 2.692-3.71 5.036-3.71z" fill="#EA4335"/>
            </svg>
            Sign in with Google
        `;
    }
}

/**
 * Sign out the current user.
 */
async function signOut() {
    try {
        await firebase.auth().signOut();
    } catch (e) {
        console.error('Firebase sign-out error:', e);
    }

    authToken = null;
    currentUser = null;
    sessionStorage.removeItem('eva_token');
    sessionStorage.removeItem('eva_user');

    showLandingPage();
}

/**
 * Get the current auth token, refreshing if needed.
 */
function getAuthToken() {
    return authToken;
}

/**
 * Make an authenticated API request.
 */
async function apiRequest(endpoint, options = {}) {
    const token = getAuthToken();
    if (!token) {
        signOut();
        throw new Error('Not authenticated');
    }

    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
        ...(options.headers || {}),
    };

    const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.apiVersion}${endpoint}`, {
        ...options,
        headers,
    });

    if (response.status === 401) {
        signOut();
        throw new Error('Session expired. Please sign in again.');
    }

    return response;
}

/**
 * Show the landing/login page.
 */
function showLandingPage() {
    document.getElementById('landing-page').classList.add('active');
    document.getElementById('chat-page').classList.remove('active');
}

/**
 * Show the chat page and update user info.
 */
function showChatPage() {
    document.getElementById('landing-page').classList.remove('active');
    document.getElementById('chat-page').classList.add('active');

    // Update user info in sidebar
    if (currentUser) {
        document.getElementById('user-name').textContent = currentUser.full_name || currentUser.username;
        const avatar = document.getElementById('user-avatar');
        if (currentUser.preferences && currentUser.preferences.picture) {
            avatar.src = currentUser.preferences.picture;
            avatar.style.display = 'block';
        }
    }

    // Focus the input
    document.getElementById('message-input').focus();
}

/**
 * Check for existing session on page load.
 */
function checkExistingSession() {
    const savedToken = sessionStorage.getItem('eva_token');
    const savedUser = sessionStorage.getItem('eva_user');

    if (savedToken && savedUser) {
        try {
            authToken = savedToken;
            currentUser = JSON.parse(savedUser);
            showChatPage();
        } catch (e) {
            sessionStorage.removeItem('eva_token');
            sessionStorage.removeItem('eva_user');
        }
    }
}

// Check for existing session on page load
document.addEventListener('DOMContentLoaded', checkExistingSession);
