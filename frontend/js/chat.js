/**
 * EVA Chat Module
 *
 * Handles chat messaging, mode switching, and UI interactions.
 */

let currentSessionId = null;
let currentMode = 'chat';
let isProcessing = false;

/**
 * Send a message to the EVA backend.
 */
async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    if (!message || isProcessing) return;

    // Clear input and reset height
    input.value = '';
    input.style.height = 'auto';
    updateSendButton();

    // Remove welcome message if present
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message to UI
    appendMessage('user', message);

    // Show typing indicator
    const typingId = showTypingIndicator();

    isProcessing = true;
    updateSendButton();

    try {
        const response = await apiRequest('/conversation/', {
            method: 'POST',
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId,
                mode: currentMode,
            }),
        });

        // Remove typing indicator
        removeTypingIndicator(typingId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || errorData.error || 'Failed to get response');
        }

        const data = await response.json();
        currentSessionId = data.session_id;

        if (data.error) {
            appendMessage('assistant', `Sorry, something went wrong: ${data.error}`);
        } else {
            appendMessage('assistant', data.response || 'No response received.');
        }

        // Show function calls if any
        if (data.function_calls && data.function_calls.length > 0) {
            for (const fc of data.function_calls) {
                appendSystemMessage(`🔧 Used tool: ${fc.name}`);
            }
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        console.error('Send message error:', error);
        appendMessage('assistant', `I'm sorry, I encountered an error: ${error.message}`);
    } finally {
        isProcessing = false;
        updateSendButton();
        input.focus();
    }
}

/**
 * Append a message to the chat UI.
 */
function appendMessage(role, content) {
    const messages = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = role === 'assistant' ? 'E' : (currentUser?.full_name?.[0] || 'U');

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    messages.appendChild(messageDiv);

    // Scroll to bottom
    messages.scrollTop = messages.scrollHeight;
}

/**
 * Append a system message (tool usage, etc.)
 */
function appendSystemMessage(content) {
    const messages = document.getElementById('messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system';
    msgDiv.style.cssText = 'align-self: center; max-width: 100%;';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.style.cssText = 'background: var(--bg-tertiary); font-size: 0.8rem; color: var(--text-muted); text-align: center;';
    contentDiv.textContent = content;

    msgDiv.appendChild(contentDiv);
    messages.appendChild(msgDiv);
    messages.scrollTop = messages.scrollHeight;
}

/**
 * Show typing indicator.
 */
function showTypingIndicator() {
    const messages = document.getElementById('messages');
    const indicator = document.createElement('div');
    const id = 'typing-' + Date.now();
    indicator.id = id;
    indicator.className = 'message assistant';
    indicator.innerHTML = `
        <div class="message-avatar">E</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    messages.appendChild(indicator);
    messages.scrollTop = messages.scrollHeight;
    return id;
}

/**
 * Remove typing indicator.
 */
function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) indicator.remove();
}

/**
 * Start a new chat session.
 */
function startNewChat() {
    currentSessionId = null;
    const messages = document.getElementById('messages');
    messages.innerHTML = `
        <div class="welcome-message">
            <div class="logo-icon">E</div>
            <h2>Hello! I'm EVA</h2>
            <p>Your personal AI assistant. How can I help you today?</p>
        </div>
    `;
    document.getElementById('message-input').focus();
}

/**
 * Switch the assistant mode.
 */
function switchMode(mode) {
    if (mode === currentMode) return;

    // Update mode
    currentMode = mode;

    // Update UI
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    const modeLabel = document.getElementById('current-mode-label');
    const modeNames = { chat: 'Chat Mode', game: 'Game Mode' };
    modeLabel.textContent = modeNames[mode] || 'Chat Mode';

    // Start new session for mode switch
    startNewChat();
}

/**
 * Toggle sidebar visibility.
 */
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('open');
    sidebar.classList.toggle('collapsed');
}

/**
 * Handle keyboard input in the message field.
 */
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/**
 * Auto-resize the textarea as user types.
 */
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    updateSendButton();
}

/**
 * Update send button state.
 */
function updateSendButton() {
    const input = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = !input.value.trim() || isProcessing;
}

// Set up input listener
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('message-input');
    if (input) {
        input.addEventListener('input', () => updateSendButton());
    }
});
