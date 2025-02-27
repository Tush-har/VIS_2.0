import os
import base64
import asyncio
import glob
from fastapi import WebSocket, APIRouter
from weakref import WeakSet
from aiofiles import open as async_open

router = APIRouter()

# This is a placeholder for storing connected clients
connected_clients = WeakSet()

async def get_base64_encoded_images(image_folder):
    images = []
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))  # Adjust the pattern if needed
    for image_path in image_paths:
        async with async_open(image_path, "rb") as image_file:
            image_data = await image_file.read()
            images.append(base64.b64encode(image_data).decode('utf-8'))
    return images

async def broadcast_images(image_folder, interval=1):
    """Broadcast the images to all connected clients every 'interval' seconds."""
    while True:
        images = await get_base64_encoded_images(image_folder)
        for base64_image in images:
            to_remove = []
            for client in connected_clients:
                try:
                    await client.send_text(base64_image)
                except Exception as e:
                    print(f"Error sending to client: {e}")
                    to_remove.append(client)
            for client in to_remove:
                connected_clients.discard(client)
        await asyncio.sleep(interval)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(0)
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        connected_clients.discard(websocket)

@router.on_event("startup")
async def startup_event():
    # Path to your image folder
    image_folder = "Images"
    # Start broadcasting the images every 5 seconds (change as needed)
    asyncio.create_task(broadcast_images(image_folder, 0))

