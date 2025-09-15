# Verité Chatbot

A chatbot project using OpenAI and Hugging Face APIs. This project demonstrates how to integrate AI models into a chatbot application using Python and Flask.

---

## ⚡ Prerequisites

- Python 3.11+
- Git
- pip

---

## 🛠 Setup Instructions

1. **Clone the repository**


git clone https://github.com/KasunChathu97/VRChatbot.git
cd VRChatbot
Create a virtual environment

bash
Copy code
python -m venv .venv
Activate it:

On Windows:

bash
Copy code
.venv\Scripts\activate
On Linux/macOS:

bash
Copy code
source .venv/bin/activate
Install Python dependencies

bash
Copy code
pip install -r requirements.txt
Set up environment variables

Copy the example file:

bash
Copy code
copy .env.example .env   # Windows
# OR
cp .env.example .env     # Linux/macOS
Open .env and add your actual API keys:

ini
Copy code
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
Do not commit your .env file to GitHub. It contains secrets.

Run the Flask application

bash
Copy code
python app.py
By default, the app runs on http://127.0.0.1:5000

