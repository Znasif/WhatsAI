# WhatsAI - Agentic GenAI Interface with Google Gemini for Blind Users

## Overview
whatsAI is an AI-powered interactive assistant designed for blind and low-vision users, leveraging Google Gemini and Meta Ray-Ban smart glasses. The system enables hands-free interaction with real-world objects using finger counting gestures to dynamically guide the AI's responses.

## Key Features
- **Finger Counting Interaction:** Uses the built-in camera to detect the number of fingers shown and triggers different AI-driven responses based on the count.
- **Meta Ray-Ban Integration:** Connects with Meta Ray-Ban smart glasses to capture real-time scenes, allowing users to interact with their surroundings seamlessly.
- **Google Gemini AI:** Processes images and audio to generate intelligent, context-aware responses about objects in the scene.
- **Hands-Free Assistance:** Provides descriptions, locations, sizes, and distances of objects, empowering blind users to navigate their environment independently.

## How It Works
1. The Meta Ray-Ban glasses capture the real-world scene and stream it to a connected PC.
2. The PC processes the video feed using OpenCV and detects the number of fingers shown.
3. The detected finger count maps to a predefined set of AI queries:
   - 1 Finger: Describe the color of objects.
   - 2 Fingers: Provide the egocentric location of objects.
   - 3 Fingers: Identify object names.
   - 4 Fingers: Estimate the size of objects.
   - 5 Fingers: Determine the distance of objects.
4. The image and selected query are sent to Google Gemini for processing.
5. The AI-generated response is converted to speech and played back to the user.

## Technology Stack
- **Hardware:** Meta Ray-Ban Smart Glasses, PC with camera
- **Software:**
  - Google Gemini API (Vision & Audio Processing)
  - OpenCV (Image Processing)
  - PyAudio (Audio Input & Output)
  - Pillow & mss (Screen Capture & Image Processing)
  - Python (Async Processing & API Integration)

## Challenges Faced
- Ensuring accurate and real-time finger counting in varying lighting conditions.
- Reducing latency in processing AI responses to provide a smooth user experience.
- Handling different camera angles and occlusions for consistent hand tracking.

## Future Improvements
- Integrate haptic feedback for enhanced interaction.
- Expand gesture-based commands beyond finger counting.
- Improve response latency with edge processing techniques.
- Develop a mobile version for standalone use with smart glasses.

## How to Run
### Installation
```
pip install google-genai opencv-python pyaudio pillow mss dotenv
```

### Setup
Ensure you have a Google Gemini API key set as an environment variable in the .env file.

### Run the Application
```
python gemini_live.py
```

## Team
Developed as a hackathon project to improve accessibility for blind and low-vision users using AI-driven real-world interactions. 🚀

