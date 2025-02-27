## LLM Voice Journal Demo

To run the server, execute the `server.py` script. Once the server is running, open your web browser and navigate to [http://localhost:9999](http://localhost:9999). Avoid using the IP address; stick to `localhost` for accessing the application.

Please note that there are a couple of UI issues that are being worked on. The server is built using Flask, with WhisperAI medium for processing and Ollama on Mistral for additional functionalities. The user interface is crafted using V0 and Cursor AI.

### Prerequisites

1. Install the required Python packages:
```sh
pip install -r requirements.txt
```

2. Install Ollama:
   - For macOS/Linux: 
     ```sh
     curl -fsSL https://ollama.ai/install.sh | sh
     ```
   - For Windows: Download from [Ollama.ai](https://ollama.ai)

3. Start Ollama and pull the Mistral model:
   ```sh
   # Start Ollama service
   ollama serve

   # In a new terminal, pull the Mistral model
   ollama pull mistral
   ```

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/AitchEm-bot/S2S.git
   ```
2. Navigate to the project directory:
   ```sh
   cd S2S
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Server

1. Make sure Ollama is running:
   ```sh
   ollama serve
   ```

2. Start the server in a new terminal:
   ```sh
   python server.py
   ```

### Accessing the Application

Open your web browser and go to [http://localhost:9999](http://localhost:9999).

### Known Issues

- UI issues are currently being addressed.

### Technologies Used

- **Backend**: Flask
- **Processing**: WhisperAI medium, Ollama on Mistral
- **Frontend**: V0, Cursor AI
- **RAG**: ChromaDB, Sentence Transformers

Feel free to contribute by submitting issues or pull requests.