// Chat page functionality
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const resetButton = document.getElementById('resetButton');

// Variable to control auto-scrolling behavior
let shouldAutoScroll = true;
let userHasScrolled = false;

// Detect when user manually scrolls
chatMessages.addEventListener('scroll', function() {
    // Calculate if user has scrolled up from bottom
    const isAtBottom = chatMessages.scrollHeight - chatMessages.clientHeight <= chatMessages.scrollTop + 50;
    
    // If not at bottom, user has manually scrolled
    userHasScrolled = !isAtBottom;
    
    // Only auto-scroll if user hasn't manually scrolled up
    shouldAutoScroll = isAtBottom;
});

async function loadChatHistory() {
    try {
        const response = await fetch('http://127.0.0.1:9999/get_chat_history');
        const data = await response.json();
        
        // Clear existing messages
        chatMessages.innerHTML = '';
        
        // Filter out messages with source="transcription"
        const visibleMessages = data.history.filter(msg => !msg.source || msg.source !== 'transcription');
        
        if (visibleMessages.length === 0) {
            // Show welcome message if there are no visible messages
            addMessageToChat("How are you feeling today? Let's have a mindful conversation.", false);
        } else {
            // Add each visible message from history
            visibleMessages.forEach(msg => {
                // Check if it's a user message (role === "user")
                const isUserMessage = msg.role === "user";
                addMessageToChat(msg.content, isUserMessage);
            });
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
        addMessageToChat('Error loading chat history');
    }
}

function addMessageToChat(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    
    // Only auto-scroll if user hasn't manually scrolled up
    if (shouldAutoScroll) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function addLoadingIndicator() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message loading-indicator';
    loadingDiv.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
    chatMessages.appendChild(loadingDiv);
    
    // Only auto-scroll if user hasn't manually scrolled up
    if (shouldAutoScroll) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    return loadingDiv;
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addMessageToChat(message, true);
    messageInput.value = '';

    // Reset scroll behavior when sending a new message
    shouldAutoScroll = true;
    userHasScrolled = false;

    // Add loading indicator
    const loadingIndicator = addLoadingIndicator();

    try {
        // Use the full URL with http://localhost:9999 instead of relative path
        const response = await fetch('http://localhost:9999/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) {
            // Remove loading indicator if there's an error
            loadingIndicator.remove();
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let receivedFirstChunk = false;
        let botMessage = '';

        // Create a new div for the bot's response
        const responseDiv = document.createElement('div');
        responseDiv.className = 'message bot-message';
        
        // Don't append the response div until we get the first chunk
        // This ensures the loading indicator is visible and in the right position

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n\n');

            for (const line of lines) {
                if (line.trim() && line.startsWith('data:')) {
                    try {
                        const jsonStr = line.substring(5).trim();
                        const data = JSON.parse(jsonStr);
                        if (data.chunk) {
                            // If this is the first chunk, append the response div and remove loading indicator
                            if (!receivedFirstChunk) {
                                chatMessages.appendChild(responseDiv);
                                loadingIndicator.remove();
                                receivedFirstChunk = true;
                                // For the first chunk, make sure it doesn't start with whitespace
                                botMessage = data.chunk.trimStart();
                            } else {
                                // For subsequent chunks, just append them
                                botMessage += data.chunk;
                            }
                            
                            responseDiv.textContent = botMessage;
                            
                            // Only auto-scroll if user hasn't manually scrolled up
                            if (shouldAutoScroll) {
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        }
                    } catch (e) {
                        console.error('Error parsing JSON:', e);
                    }
                }
            }
        }
        
        // If we never received any chunks, remove the loading indicator
        // and show an empty response
        if (!receivedFirstChunk) {
            loadingIndicator.remove();
            chatMessages.appendChild(responseDiv);
            responseDiv.textContent = "No response received.";
            
            // Only auto-scroll if user hasn't manually scrolled up
            if (shouldAutoScroll) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
    } catch (error) {
        console.error('Error:', error);
        // Make sure to remove loading indicator if it's still there
        if (loadingIndicator.parentNode) {
            loadingIndicator.remove();
        }
        addMessageToChat(`Error: ${error.message}`, false);
    }
}

async function resetContext() {
    try {
        const response = await fetch('http://localhost:9999/reset_context', {
            method: 'POST'
        });
        const data = await response.json();
        
        // Clear chat messages and add reset confirmation
        chatMessages.innerHTML = '';
        addMessageToChat("Context has been reset. How can I help you?", false);
        
    } catch (error) {
        console.error('Error:', error);
        addMessageToChat('Error: Could not reset context');
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
resetButton.addEventListener('click', resetContext);

// Load chat history when page loads
document.addEventListener('DOMContentLoaded', loadChatHistory); 