import os
from fastapi.responses import JSONResponse
from fastapi import Request,HTTPException
from fastapi import HTTPException, File, UploadFile, Form, HTTPException, APIRouter
from pydantic import BaseModel,Field
from pymongo import MongoClient
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pymongo.errors import DuplicateKeyError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pymongo import MongoClient
import json
from typing import List




# mongo db connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
# FastAPI application instance
router = APIRouter()

# MongoDB client
client = MongoClient(MONGODB_URL)
# insert new collection
db = client['config_file']
config_collection = db['camera_config']

# to fetch models
db1 = client['model_training_db']
trained_models_collection = db1['trained_models']




# class CameraConfig(BaseModel):
#     camera_id: str
#     ok_classes: List[str]  # List of OK class names
#     ng_classes: List[str]  # List of NG class names
#     cam_settings: List[float]  # List of camera setting values # List of float values for camera settings
#     model_path: str
#     color_classes: str  # New field for color mappings


class CameraConfig(BaseModel):
    camera_id: str
    ok_classes: List[str]  # List of OK class names
    ng_classes: List[str]  # List of NG class names
    cam_settings: List[float]  # List of float values for camera settings
    model_path: str
    color_classes: str = None  # Will be auto-generated

@router.post("/upload_data/")
def upload_data(config: CameraConfig):
    try:
        # Check if the camera_id already exists
        existing_config = config_collection.find_one({"camera_id": config.camera_id})
        if existing_config:
            return {"message": "Camera ID already exists"}

        # Generate color_classes in correct JSON string format
        color_classes_dict = {
            "red": config.ng_classes,   # NG Classes → Red
            "green": config.ok_classes  # OK Classes → Green
        }
        color_classes_str = json.dumps(color_classes_dict, ensure_ascii=False)  # Proper JSON formatting

        # Prepare data for MongoDB
        data = config.dict()
        data["color_classes"] = color_classes_str  # Store as a string

        # Insert into MongoDB
        result = config_collection.insert_one(data)

        return {
            "message": "Camera config uploaded successfully",
            "inserted_id": str(result.inserted_id)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# Get all camera ids from collections
@router.get("/all_cam_data/")
def get_all_camera_data():
    try:
        camera_data = list(config_collection.find({}, {"_id": 0}))
        if camera_data:
            return {"camera_data": camera_data}
        else:
            return {"No data exist in database!"}
    except Exception as e:
        return {"error": str(e)}
    

# Delete collection by cam_id
@router.delete("/delete_camera/{camera_id}")
def delete_camera(camera_id: str):
    try:
        result = config_collection.delete_one({"camera_id": camera_id})
        if result.deleted_count == 0:
            return{"Data not exist for this camera id!"}
        return {"message": "Camera config deleted successfully"}
    except Exception as e:
        return {"message" : "No data exist in database"}


# Update collection by cam_id
# @router.put("/update_camera/{camera_id}")
# def update_camera(camera_id: str, config: CameraConfig):
#     try:
#         existing_config = config_collection.find_one({"camera_id": camera_id})

#         if not existing_config:
#             return{"message" : "camera id not found!"}
        
#         update_data = {"$set": config.model_dump()}
#         config_collection.update_one({"camera_id": camera_id}, update_data)
        
#         return {"message": "Camera config updated successfully"}
#     except Exception as e:
#         return {"message" : "data not exist"}


class ModelUpdate(BaseModel):
    camera_id: str
    model_path: str
    cam_settings: List[float] = Field(default=None)
    ok_classes: list[str] = Field(default=None)
    ng_classes: list[str] = Field(default=None)

@router.put("/update_model/")
async def update_model(data: ModelUpdate):
    try:
        # Find existing document
        existing_doc = config_collection.find_one({"camera_id": data.camera_id})
        if not existing_doc:
            return {"message":"Camera not exists!"}

        update_fields = {"model_path": data.model_path}  # Always update model_path
        if data.cam_settings is not None:
            update_fields = {"cam_settings": data.cam_settings} #update camera settings if given
        else:
            data.cam_settings = existing_doc.get("cam_setting",[])

        # Only update ok_classes & ng_classes if they are provided
        if data.ok_classes is not None:
            update_fields["ok_classes"] = data.ok_classes
        else:
            data.ok_classes = existing_doc.get("ok_classes", [])

        if data.ng_classes is not None:
            update_fields["ng_classes"] = data.ng_classes
        else:
            data.ng_classes = existing_doc.get("ng_classes", [])

        # If either ok_classes or ng_classes were provided, update color_classes
        if data.ok_classes or data.ng_classes:
            color_classes_dict = {
                "green": data.ok_classes if data.ok_classes else [],
                "red": data.ng_classes if data.ng_classes else []
            }
            update_fields["color_classes"] = json.dumps(color_classes_dict)  # ✅ Convert dict to string

        # Perform update
        result = config_collection.update_one(
            {"camera_id": data.camera_id},
            {"$set": update_fields}
        )

        return {
            "message": "Model path and optional classes updated successfully!",
            "updated_fields": update_fields
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top_models/")
async def get_top_models():
    try:
        top_models = trained_models_collection.aggregate([
    {
        "$sort": {"mAP": -1}  # Sort by highest mAP first
    },
    {
        "$group": {
            "_id": "$project_name",  # Group by project_name
            "top_models": {
                "$push": {
                    "project_name": "$project_name",
                    "best_model_path": "$best_model_path",
                    "mAP": "$mAP"
                }
            }
        }
    },
    {
        "$project": {
            "_id": 0,
            "project_name": "$_id",
            "top_models": {"$slice": ["$top_models", 3]}  # Keep only top 3 per project
        }
    }
])
        top_models_list = list(top_models)

        return top_models_list
    
    except Exception as e:
        return {"message":"No model exists!"}
