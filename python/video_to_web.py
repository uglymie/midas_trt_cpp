import cv2
import asyncio
import websockets
import base64
import numpy as np

# Read the video file
video_path = "D://video/sample.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize the websockets server
async def video_server(websocket, path):
    while True:
        # Read the video frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image data to Base64 encoding
        _, img_encode = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encode.tobytes()).decode('utf-8')

        # Send the Base64 encoded image data to the client
        await websocket.send(img_base64)

async def main():
    # Set up the websockets server parameters
    # server = await websockets.serve(video_server, 'localhost', 8765)
    server = await websockets.serve(video_server, '168.168.0.17', 8765)

    # Start the event loop
    await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())
