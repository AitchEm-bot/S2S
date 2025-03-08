// DOM Elements
let canvas, ctx, statusText, debugInfo;

// Animation variables
let animationId;
let audioContext;
let analyser;
let dataArray;
let source;
let recognition;
let currentState = 'listening'; // listening, processing, speaking
let circleProgress = 0;
let lastUpdateTime = Date.now();
let speechSynthesisUtterance = null;

// Debug function
function updateDebug(message) {
    if (debugInfo) {
        debugInfo.textContent = message;
    }
}

// Set canvas dimensions
function resizeCanvas() {
    if (!canvas || !ctx) return;
    
    // Set canvas to window dimensions exactly
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    updateDebug(`Canvas size: ${canvas.width}x${canvas.height}`);
    
    // Redraw visualization
    if (currentState) {
        drawVisualization();
    }
}

// Initialize speech recognition
function initSpeechRecognition() {
    try {
        window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        
        recognition.onstart = () => {
            changeState('listening');
            updateDebug("Speech recognition started");
        };
        
        recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map(result => result[0])
                .map(result => result.transcript)
                .join('');
            
            if (event.results[0].isFinal) {
                processUserSpeech(transcript);
            }
        };
        
        recognition.onend = () => {
            if (currentState === 'listening') {
                recognition.start();
            }
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error', event.error);
            updateDebug(`Speech error: ${event.error}`);
            
            if (event.error === 'no-speech') {
                // Restart if no speech detected
                if (currentState === 'listening') {
                    recognition.start();
                }
            } else {
                setTimeout(() => {
                    changeState('listening');
                    recognition.start();
                }, 3000);
            }
        };
        
        recognition.start();
        updateDebug("Speech recognition initialized");
    } catch (err) {
        console.error("Speech recognition init error:", err);
        updateDebug(`Speech init error: ${err.message}`);
        // Continue with visualization even if speech recognition fails
        startVisualization();
    }
}

// Initialize audio context for visualization
function initAudioContext() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);
        
        updateDebug("Audio context created");
        
        // Get microphone input
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);
                
                updateDebug("Microphone connected");
                
                // Start visualization
                startVisualization();
                
                // Start speech recognition
                initSpeechRecognition();
            })
            .catch(err => {
                console.error('Error accessing microphone:', err);
                updateDebug(`Mic error: ${err.message}`);
                
                // Start visualization anyway with default animation
                startVisualization();
                
                // Add click listener to retry
                document.addEventListener('click', retryMicrophoneAccess, { once: true });
            });
    } catch (err) {
        console.error("Audio context init error:", err);
        updateDebug(`Audio init error: ${err.message}`);
        // Start visualization anyway with default animation
        startVisualization();
    }
}

// Retry microphone access
function retryMicrophoneAccess() {
    initAudioContext();
}

// Process user speech and get AI response
function processUserSpeech(text) {
    if (!text.trim()) return;
    
    changeState('processing');
    
    // Send the speech to the server for processing
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: text,
            stream: false
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            speakResponse(data.response);
        } else {
            // Fallback response if no valid response
            speakResponse("I'm sorry, I couldn't process that properly. Could you try again?");
        }
    })
    .catch(error => {
        console.error('Error:', error);
        speakResponse("I'm having trouble connecting to the server. Please try again in a moment.");
    });
}

// Speak AI response using speech synthesis
function speakResponse(text) {
    changeState('speaking');
    
    // Cancel any existing speech
    if (speechSynthesisUtterance) {
        speechSynthesis.cancel();
    }
    
    speechSynthesisUtterance = new SpeechSynthesisUtterance(text);
    speechSynthesisUtterance.rate = 1.0;
    speechSynthesisUtterance.pitch = 1.0;
    speechSynthesisUtterance.volume = 1.0;
    
    speechSynthesisUtterance.onend = () => {
        speechSynthesisUtterance = null;
        changeState('listening');
    };
    
    speechSynthesis.speak(speechSynthesisUtterance);
}

// Interrupt AI speech
function interruptSpeech() {
    if (currentState === 'speaking' && speechSynthesis.speaking) {
        speechSynthesis.cancel();
        speechSynthesisUtterance = null;
        changeState('listening');
        return true;
    }
    return false;
}

// Change visualization state
function changeState(newState) {
    currentState = newState;
    updateDebug(`State: ${newState}`);
}

// Start visualization loop
function startVisualization() {
    // Start animation loop
    lastUpdateTime = Date.now();
    animationLoop();
    updateDebug("Animation started");
}

// Main animation loop
function animationLoop() {
    animationId = requestAnimationFrame(animationLoop);
    drawVisualization();
}

// Draw visualization based on current state
function drawVisualization() {
    if (!canvas || !ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Update time
    const now = Date.now();
    const deltaTime = (now - lastUpdateTime) / 1000;
    lastUpdateTime = now;
    
    // Get audio data if available
    if (analyser && currentState === 'listening') {
        analyser.getByteFrequencyData(dataArray);
    }
    
    // Update visualization based on state
    switch (currentState) {
        case 'listening':
            drawSoundWave(width, height);
            break;
        case 'processing':
            drawLoadingCircle(centerX, centerY, Math.min(width, height) * 0.15, deltaTime);
            break;
        case 'speaking':
            drawSpeakingCircle(centerX, centerY, Math.min(width, height) * 0.15, deltaTime);
            break;
    }
}

// Draw horizontal sound wave
function drawSoundWave(width, height) {
    const centerY = height / 2;
    const waveHeight = height * 0.15; // Reduced height for subtlety
    
    // Draw the wave (no base line for cleaner look)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 51, 102, 0.3)'; // Even more transparent
    ctx.lineWidth = 2; // Thin line
    
    const segments = 150; // More segments for smoother curve
    const segmentWidth = width / segments;
    
    // If we have audio data, use it; otherwise, create a default animation
    const hasAudioData = analyser && dataArray && dataArray.some(val => val > 0);
    
    for (let i = 0; i <= segments; i++) {
        const x = i * segmentWidth;
        let y;
        
        if (hasAudioData) {
            // Use real audio data but make it more subtle
            const freqIndex = Math.floor(i / segments * dataArray.length);
            const amplitude = dataArray[freqIndex] / 255 * waveHeight * 0.6 || 0; // Further reduced amplitude
            y = centerY + amplitude * Math.sin(i / 5 + Date.now() / 1000); // Even slower movement
        } else {
            // Create an even more subtle default wave animation
            const time = Date.now() / 2000; // Much slower time factor
            const amplitude = waveHeight * 0.15; // Further reduced amplitude
            
            // Combine multiple sine waves for a very gentle, natural look
            y = centerY + 
                amplitude * Math.sin(i / 25 + time * 0.8) * 0.5 +
                amplitude * Math.sin(i / 15 - time * 0.6) * 0.3 +
                amplitude * Math.sin(i / 40 + time * 0.4) * 0.2;
        }
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
    
    // Add extremely subtle glow effect
    let avgIntensity = 0.1; // Further reduced intensity
    if (hasAudioData) {
        avgIntensity = Array.from(dataArray).reduce((sum, val) => sum + val, 0) / dataArray.length / 255;
        avgIntensity = Math.max(avgIntensity, 0.05) * 0.15; // Scale down intensity even more
    } else {
        avgIntensity = 0.05 + Math.sin(Date.now() / 1500) * 0.03;
    }
    
    // Create very subtle glow along the wave
    const gradient = ctx.createLinearGradient(0, centerY - waveHeight, 0, centerY + waveHeight);
    gradient.addColorStop(0, `rgba(102, 179, 255, 0)`);
    gradient.addColorStop(0.5, `rgba(102, 179, 255, ${0.05 + avgIntensity * 0.1})`); // Even more subtle glow
    gradient.addColorStop(1, `rgba(102, 179, 255, 0)`);
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, centerY - waveHeight, width, waveHeight * 2);
}

// Draw loading animation while processing
function drawLoadingCircle(centerX, centerY, radius, deltaTime) {
    // Update loading progress much slower
    circleProgress += deltaTime * 0.8; // Much slower animation
    
    // Draw subtle pulsing circle
    const pulseScale = 1 + Math.sin(circleProgress * 1.2) * 0.02; // Reduced pulse and slower
    
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 51, 102, 0.3)'; // More transparent
    ctx.lineWidth = 1.5; // Thinner line
    ctx.arc(centerX, centerY, radius * pulseScale, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw loading dots (more subtle and slower)
    const dotCount = 5; // Fewer dots
    const dotRadius = radius * 0.04; // Smaller dots
    
    for (let i = 0; i < dotCount; i++) {
        const angle = (i / dotCount) * Math.PI * 2 + circleProgress * 0.7; // Slower rotation
        const x = centerX + Math.cos(angle) * radius * 1.1;
        const y = centerY + Math.sin(angle) * radius * 1.1;
        const opacity = (Math.sin(angle * 2 + circleProgress * 3) + 1) / 2 * 0.3; // Reduced opacity
        
        ctx.beginPath();
        ctx.fillStyle = `rgba(0, 51, 102, ${opacity})`;
        ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
        ctx.fill();
    }
    
    // Add subtle glow effect
    ctx.beginPath();
    const gradient = ctx.createRadialGradient(centerX, centerY, radius * 0.8, centerX, centerY, radius * 1.3);
    gradient.addColorStop(0, 'rgba(102, 179, 255, 0.1)'); // Reduced opacity
    gradient.addColorStop(1, 'rgba(102, 179, 255, 0)');
    ctx.fillStyle = gradient;
    ctx.arc(centerX, centerY, radius * 1.3, 0, Math.PI * 2);
    ctx.fill();
}

// Draw speaking animation as a circle with much slower vibrations
function drawSpeakingCircle(centerX, centerY, radius, deltaTime) {
    // Update speaking animation much slower
    circleProgress += deltaTime * 0.7; // Much slower animation
    
    // Draw ripple effect (more subtle and slower)
    const rippleCount = 2; // Fewer ripples
    
    for (let i = 0; i < rippleCount; i++) {
        const rippleProgress = ((circleProgress * 0.6 + i / rippleCount) % 1); // Slower ripples
        const rippleRadius = radius * (1 + rippleProgress * 0.3); // Reduced expansion
        const opacity = (1 - rippleProgress) * 0.25; // Reduced opacity
        
        ctx.beginPath();
        ctx.strokeStyle = `rgba(0, 51, 102, ${opacity})`;
        ctx.lineWidth = 1; // Even thinner line
        ctx.arc(centerX, centerY, rippleRadius, 0, Math.PI * 2);
        ctx.stroke();
    }
    
    // Draw main circle with very subtle wave effect
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 51, 102, 0.4)'; // More transparent
    ctx.lineWidth = 1.5; // Thinner line
    
    const points = 100;
    const angleStep = (Math.PI * 2) / points;
    const waveAmplitude = 0.02; // Reduced amplitude
    const waveFrequency = 6; // Lower frequency for slower waves
    
    for (let i = 0; i <= points; i++) {
        const angle = i * angleStep;
        const waveOffset = Math.sin(angle * waveFrequency + circleProgress * 1.5) * waveAmplitude; // Slower movement
        
        const x = centerX + Math.cos(angle) * radius * (1 + waveOffset);
        const y = centerY + Math.sin(angle) * radius * (1 + waveOffset);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.closePath();
    ctx.stroke();
    
    // Add subtle glow effect
    ctx.beginPath();
    const pulseIntensity = (Math.sin(circleProgress * 1.5) + 1) / 2 * 0.1 + 0.05; // Reduced intensity and slower
    const gradient = ctx.createRadialGradient(centerX, centerY, radius * 0.8, centerX, centerY, radius * 1.3);
    gradient.addColorStop(0, `rgba(102, 179, 255, ${pulseIntensity})`);
    gradient.addColorStop(1, 'rgba(102, 179, 255, 0)');
    ctx.fillStyle = gradient;
    ctx.arc(centerX, centerY, radius * 1.3, 0, Math.PI * 2);
    ctx.fill();
}

// Check if a point is inside the circle
function isPointInCircle(x, y, centerX, centerY, radius) {
    const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
    return distance <= radius * 1.5; // Larger hit area for easier clicking
}

// Initialize everything when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    canvas = document.getElementById('visualizer');
    ctx = canvas.getContext('2d');
    statusText = document.getElementById('status-text');
    debugInfo = document.getElementById('debug-info');
    
    // Initialize canvas
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Handle canvas click
    canvas.addEventListener('click', (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = Math.min(canvas.width, canvas.height) * 0.15;
        
        // If in speaking state and click is inside the circle, interrupt speech
        if (currentState === 'speaking' && isPointInCircle(x, y, centerX, centerY, radius)) {
            interruptSpeech();
        }
        
        // Resume audio context if suspended (needed for some browsers)
        if (audioContext && audioContext.state === 'suspended') {
            audioContext.resume();
        }
    });
    
    // Initialize audio context
    initAudioContext();
    
    // Start visualization immediately with default animation
    startVisualization();
}); 