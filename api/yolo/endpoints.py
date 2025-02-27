# import os
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# from fastapi import APIRouter
# from config import collection

# inspection = APIRouter()

# def inference_model(image_paths):
#     model = YOLO("yolov8m.pt")
#     for image in os.listdir(image_paths):
#         object_count = {}
#         image_path = image_paths + '/' + image
#         results = model.predict(image_path)
#         result = results[0]
#         box = result.boxes[0]

#         cords = box.xyxy[0].tolist()
#         class_id = box.cls[0].item()
#         print(result.names)
#         cords = box.xyxy[0].tolist()
#         cords = [round(x) for x in cords]
#         class_id = result.names[box.cls[0].item()]

#         for box in result.boxes:
#             class_id = result.names[box.cls[0].item()]
#             cords = box.xyxy[0].tolist()
#             cords = [round(x) for x in cords]
#             object_count[class_id] = object_count.get(class_id, 0) + 1

#         output_path = f"C:/Users/utkrisht.dutta/Documents/api/output_images/annotated_{image_path.split('/')[-1]}"
#         img_array = result.plot()[:,:,::-1]
#         img = Image.fromarray(np.uint8(img_array))
#         img.save(output_path)
#         image_dict = {"image_path": output_path, "count": object_count}
#         result = collection.insert_one(image_dict)

#     return

# @inspection.post('/start_inpection')
# async def start_inspection(file: dict):
#     try:
#         pred_classes = ['Good', 'Def']
#         inference_model(file['input_path'])
#         return 
#     except Exception as e:
#         return {"error": str(e)}
    
