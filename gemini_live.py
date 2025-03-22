# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GEMINI_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python agentic_gemini.py
```

The script uses camera by default to detect hand gestures and count fingers.
"""

import asyncio
import base64
import io
import os
import sys
import traceback
import time
import json
from dotenv import load_dotenv
import cv2
import pyaudio
import PIL.Image
import mss
import argparse

from google import genai

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Models
GEMINI_AUDIO_MODEL = "models/gemini-2.0-flash-exp"  # For audio response

DEFAULT_MODE = "camera"
DEFAULT_TEXT_PROMPT = """
You are a finger counting system. 
Your task is to count the number of fingers being shown in the image.
Only count clearly visible and extended fingers in the scene.
Return ONLY the number as an integer without any explanation.
If you can't see any hands or fingers, return 0.
If you see more than 5 fingers, return the exact count.
"""

QUESTIONS = [
    "What's the name of the object that the person is pointing to?",  # 3 fingers
    "What's the color of the object being touched in the scene?",  # 1 finger
    "What's the egocentric location of different objects in the scene assuming this is a POV camera?",  # 2 fingers
    "What's the size of object in the scene being pointed to?",  # 4 fingers
    "What's the distance of object in the scene being pointed to?"  # 5 fingers
]

# Build the full audio prompt based on finger count
def build_audio_prompt(finger_count):
    if finger_count == 0:
        return "Keep all responses extremely short and to the point. Continue with the last question but based on this new scene."
    elif finger_count > 5:
        # More than 5 fingers, use question 3
        return "Keep all responses extremely short and to the point. " + QUESTIONS[0]
    else:
        # 1-5 fingers, use corresponding question
        return "Keep all responses extremely short and to the point. " + QUESTIONS[finger_count - 1]

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'), http_options={"api_version": "v1alpha"})

# While Gemini 2.0 Flash is in experimental preview mode, only one of AUDIO or
# TEXT may be passed here.
CONFIG = {"response_modalities": ["AUDIO"]}

pya = pyaudio.PyAudio()


class AgenticAudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.is_model_responding = False
        self.audio_output_complete = asyncio.Event()
        self.audio_output_complete.set()  # Start with it set so we can send the first prompt
        self.last_finger_count = 0
        self.last_question = QUESTIONS[2]  # Default to question 3
        self.frame_queue = asyncio.Queue(maxsize=1)  # Queue to hold the latest frame
        self.text_client = client  # For text/vision analysis
        self.send_prompt_lock = asyncio.Lock()  # Lock to prevent multiple prompts being sent at once

    async def process_frame_with_gemini(self):
        """Process frames with Gemini text model to count fingers"""
        while True:
            try:
                # Get the latest frame from the queue, don't wait if empty
                try:
                    frame = self.frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    if self.is_model_responding:
                        continue
                    # Create the content parts for the Gemini API request
                    content_parts = [
                        {"text": DEFAULT_TEXT_PROMPT},
                        {"inline_data": frame}
                    ]
                    
                    # Make the API call to the text model
                    response = self.text_client.models.generate_content(
                        model="gemini-2.0-flash-lite",
                        contents=content_parts
                    )
                    
                    # Extract the finger count from the response
                    try:
                        finger_count = int(response.text.strip())
                        if finger_count != self.last_finger_count:
                            self.last_finger_count = finger_count
                            print(f"Detected {finger_count} fingers")
                            
                            # Build the appropriate audio prompt
                            if finger_count == 0:
                                prompt = "Continue with the last question but based on this new scene."
                            elif finger_count > 5:
                                self.last_question = QUESTIONS[2]  # Question 3 for >5 fingers
                                prompt = self.last_question
                            else:
                                self.last_question = QUESTIONS[finger_count - 1]
                                prompt = self.last_question
                            
                            # Wait for previous audio output to complete
                            #await self.audio_output_complete.wait()
                            
                            # Check again if we're already responding (double check)
                            if not self.is_model_responding:
                                print("Model is not responding, sending prompt.")
                                self.is_model_responding = True
                                async with self.send_prompt_lock:
                                    print(f"Sending prompt: {prompt}")
                                    self.audio_output_complete.clear()  # Clear before sending
                                    await self.session.send(input=prompt, end_of_turn=True)
                            else:
                                print("Model is already responding, skipping prompt.")
                            
                    except ValueError:
                        print(f"Error parsing finger count: {response.text}")
                except Exception as e:
                    print(f"Error calling Gemini text API: {e}")
            
            except Exception as e:
                print(f"Error in processing frame: {e}")
            
            # Wait before processing the next frame
            await asyncio.sleep(0.5)  # Slightly increased from 0.1 to reduce CPU usage

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {
            "audio_frame": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()},
            "text_frame": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
        }

    async def get_screen(self):
        while True:
            frames = await asyncio.to_thread(self._get_screen)
            if frames is None:
                break
            await self.out_queue.put(frames["audio_frame"])
            
            # Update the frame for text processing (overwrite if not consumed yet)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self.frame_queue.put(frames["text_frame"])

            await asyncio.sleep(0.1)  # Faster frame capture

    def _get_frame(self, cap):
        # Read the frame
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        
        # Return both the frame for the audio session and for Gemini text analysis
        return {
            "audio_frame": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()},
            "text_frame": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
        }

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frames = await asyncio.to_thread(self._get_frame, cap)
            if frames is None:
                break

            # Put the frame in both queues
            await self.out_queue.put(frames["audio_frame"])
            
            # Update the frame for text processing (overwrite if not consumed yet)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self.frame_queue.put(frames["text_frame"])

            await asyncio.sleep(0.1)  # Faster frame capture

        # Release the VideoCapture object
        cap.release()

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """Background task to read from the websocket and write pcm chunks to the output queue"""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()
            
            # Mark that the model has finished responding
            self.is_model_responding = False
            print("\n[Response complete]")
            
            # Signal that audio playback is complete
            self.audio_output_complete.set()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            print(f"Starting with video mode: {self.video_mode}")
            print("Using camera to detect finger count and select questions")
            print("Press Ctrl+C to exit")
            
            async with (
                client.aio.live.connect(model=GEMINI_AUDIO_MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Add the task for processing frames with Gemini text
                tg.create_task(self.process_frame_with_gemini())
                
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                
                tg.create_task(self.get_screen())  # Always use camera mode

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # Create a dummy task to keep the program running
                dummy_task = tg.create_task(asyncio.sleep(float('inf')))
                await dummy_task

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if hasattr(self, 'audio_stream'):
                self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    try:
        main = AgenticAudioLoop(video_mode="screen")
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)