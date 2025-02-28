// Record page functionality
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
const recordButton = document.getElementById('recordButton');
const background = document.getElementById('background');
let rotationAngle = 135;
let colorPhase = 0;

function animateGradient() {
    rotationAngle = (rotationAngle + 0.05) % 360;
    colorPhase = (colorPhase + 0.005) % (Math.PI * 2);
    const color1 = `hsl(210, ${90 + Math.sin(colorPhase) * 10}%, ${95 + Math.sin(colorPhase) * 5}%)`;
    const color2 = `hsl(215, ${80 + Math.cos(colorPhase) * 10}%, ${85 + Math.cos(colorPhase) * 5}%)`;
    const color3 = `hsl(220, ${70 + Math.sin(colorPhase + Math.PI/2) * 10}%, ${75 + Math.sin(colorPhase + Math.PI/2) * 5}%)`;

    background.style.background = `linear-gradient(${rotationAngle}deg, ${color1}, ${color2}, ${color3})`;
    updateButtonColor(color1, color2, color3);
    
    if (isRecording) {
        requestAnimationFrame(animateGradient);
    }
}

function updateButtonColor(color1, color2, color3) {
    if (isRecording) {
        const gradientColor = `linear-gradient(${rotationAngle}deg, ${color1}, ${color2}, ${color3})`;
        recordButton.style.background = gradientColor;
    }
}

async function toggleRecording() {
    isRecording = !isRecording;
    
    if (isRecording) {
        // Start Recording
        try {
            let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                let audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                let formData = new FormData();
                formData.append("audio", audioBlob, "recording.wav");

                // Send to backend
                try {
                    const response = await fetch("http://127.0.0.1:9999/listen_audio", {
                        method: "POST",
                        body: formData
                    });
                    const data = await response.json();
                    console.log("Transcription:", data.transcription);
                } catch (error) {
                    console.error("Error:", error);
                }

                audioChunks = [];
            };

            mediaRecorder.start();
            recordButton.style.backgroundColor = 'transparent';
            recordButton.querySelector('svg').style.fill = '#ffffff';
            animateGradient();
        } catch (error) {
            console.error("Error accessing microphone:", error);
            isRecording = false;
            alert("Could not access microphone. Please check permissions.");
        }
    } else {
        // Stop Recording
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        recordButton.style.backgroundColor = '#ffffff';
        recordButton.querySelector('svg').style.fill = '#003366';
        background.style.background = 'linear-gradient(135deg, #e6f2ff, #b3d9ff, #66b3ff)';
    }
}

// Event listeners
recordButton.addEventListener('click', toggleRecording); 