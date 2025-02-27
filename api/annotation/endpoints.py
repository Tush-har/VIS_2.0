import os
import shutil
import signal
import csv
import threading
import time
import asyncio
import httpx
import zipfile
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import base64
import time
from fastapi import FastAPI, HTTPException, Query
import ultralytics
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from fastapi import Request,HTTPException
from fastapi import HTTPException, File, UploadFile, Form, HTTPException, APIRouter
from pydantic import BaseModel
from pymongo import MongoClient
# from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
from fastapi.responses import JSONResponse
from bson import ObjectId
from typing import List, Optional
from datetime import datetime
import pandas as pd


# Load environment variables from a .env file
load_dotenv()

# Constants and paths
BASE_DIR = Path(os.getenv("BASE_DIR", r"C:\SPRL\api\Project"))
TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", r"C:\SPRL\api\Training_data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", r"C:\SPRL\api\Models"))
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

# Hardcoded base directory for API to read results from CSV (read_results)
CSV_DIR = r"C:\SPRL\api\Models"
TEMP_DIR = Path(r"C:\SPRLapi\temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI application instance
router = APIRouter()

# MongoDB client
client = MongoClient(MONGODB_URL)
db = client['model_training_db']
current_training_collection = db['current_training']
trained_models_collection = db['trained_models']

# MongoDB connection details
MONGO_URL = "mongodb://localhost:27017"
DATABASE = "mydatabase"
COLLECTION = "configurations"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Box(BaseModel):
    type: str
    x: float
    y: float
    w: float
    h: float
    highlighted: bool
    editingLabels: bool
    color: str
    cls: str
    id: str



class PixelSize(BaseModel):
    w: int
    h: int

class Payload(BaseModel):
    src: str
    name: str
    pixelSize: PixelSize
    regions: List[Box]



@router.post("/upload_data/")
async def upload_file(file: UploadFile = File(...), project_name: str = Form(...)):
    project_dir = BASE_DIR / project_name

    # Ensure the project directory exists
    os.makedirs(project_dir, exist_ok=True)

    # Save the uploaded zip file to a temporary location
    temp_zip_path = project_dir / file.filename
    with open(temp_zip_path, "wb") as buffer:
        buffer.write(await file.read())

    # Extract the zip file into the project directory
    try:
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(project_dir)
    except zipfile.BadZipFile:
        return JSONResponse(status_code=400, content={"message": "Invalid zip file"})

    # Remove the temporary zip file after extraction
    os.remove(temp_zip_path)

    return {"message": f"File '{file.filename}' successfully uploaded and extracted to '{project_name}'"}


def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return None  # Return None if any error occurs

@router.get("/get_all_project")
def get_all_projects():
    projects = {}
    
    if not os.path.exists(BASE_DIR):
        return {"error": "Base directory not found"}

    for project in os.listdir(BASE_DIR):
        project_path = os.path.join(BASE_DIR, project)

        if os.path.isdir(project_path):  # Only process directories
            images = [f for f in os.listdir(project_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            xml_files = {f.replace('.xml', '') for f in os.listdir(project_path) if f.lower().endswith('.xml')}

            image_details = []
            for img in images:
                img_name, _ = os.path.splitext(img)  # Remove extension
                has_xml = img_name in xml_files
                image_details.append({"image": img, "xml_present": has_xml})

            # Get Base64 of the first image if available
            first_image_path = os.path.join(project_path, images[0]) if images else None
            image_url = encode_image_to_base64(first_image_path) if first_image_path else None

            projects[project] = {
                "image_count": len(images),
                "xml_count": len(xml_files),
                "image_url": image_url,  # Base64-encoded first image
                "images": image_details
            }

    return {"projects": projects}


class Project(BaseModel):
    project_name: str

@router.post("/create_project/") 
async def create_project(request: Request, project: Project):
    try:
        project_name = project.project_name
        project_dir = BASE_DIR / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        return {"message": f"Project '{project_name}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/get_project_data/")
async def get_project_data(request: Request, project: Project):
    try:
        project_name = project.project_name
        project_dir = BASE_DIR / project_name
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

        # Get all image and text file names in the project directory
        image_files = []
        for f in project_dir.glob("*"):
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                try:
                    with open(f, "rb") as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode("utf-8")
                        image_files.append({"name": f.name, "data": img_base64})
                        # Debug print for each image file processed
                        # print(f"Processed image: {f.name}, Base64 length: {len(img_base64)}")
                except Exception as img_e:
                    print(f"Failed to process image {f.name}: {img_e}")

        # Debug print to check image_files content
        # print("Image Files:", image_files)

        txt_files = [f.name for f in project_dir.glob("*.txt")]

        # Construct the response data
        response_data = {
            "images": image_files,
            "text_files": txt_files
        }

        # Debug print to check the final response content
        # print("Response Data:", response_data)

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    # .........................................................


class PixelSize(BaseModel):
    w: int
    h: int

class Region(BaseModel):
    type: str
    x: float
    y: float
    w: float
    h: float
    highlighted: bool
    editingLabels: bool
    color: str
    cls: str
    id: str

class Payload(BaseModel):
    pixelSize: PixelSize
    regions: List[Region]
    image_name: str
    project_name: str

def create_xml_annotation(payload: Payload) -> str:
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"
    filename = ET.SubElement(annotation, "filename")
    filename.text = payload.image_name

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(payload.pixelSize.w)
    height = ET.SubElement(size, "height")
    height.text = str(payload.pixelSize.h)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"  # Assuming the depth is 3 (for RGB images)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    
    for region in payload.regions:
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = region.cls
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        # Ensure all expected fields are present
        region_id = ET.SubElement(obj, "id")
        region_id.text = region.id
        
        color = ET.SubElement(obj, "color")
        color.text = region.color
        
        highlighted = ET.SubElement(obj, "highlighted")
        highlighted.text = "true" if region.highlighted else "false"
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(region.x * payload.pixelSize.w))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(region.y * payload.pixelSize.h))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int((region.x + region.w) * payload.pixelSize.w))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int((region.y + region.h) * payload.pixelSize.h))
    
    return ET.tostring(annotation, encoding="unicode")

def save_annotation_to_file(annotation_xml: str, filename: str):
    project_dir = BASE_DIR / filename
    print(project_dir)
    with open(project_dir, "w") as file:
        file.write(annotation_xml)

@router.post("/upload_data/save_annotation")
async def save_annotation(payload: Payload):
    try:
        xml_annotation = create_xml_annotation(payload)
        annotation_path = BASE_DIR / payload.project_name / f"{payload.image_name}.xml"
        
        # Ensure directory exists
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(annotation_path, "w") as file:
            file.write(xml_annotation)
        
        return {"xml": xml_annotation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {str(e)}")


# get annotations API

class AnnotationRequest(BaseModel):
    project_name: str
    image_name: str


@router.post("/get_annotation")
async def get_annotation(request: AnnotationRequest):
    try:
        project_name = request.project_name
        image_name = request.image_name
        annotation_path = BASE_DIR / project_name / f"{image_name}.xml"

        if not annotation_path.exists():
            return JSONResponse(status_code=404, content={"message": f"File not found for project: {project_name}, image: {image_name}"})

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        pixelSize = {"w": width, "h": height}

        regions = []
        for obj in root.findall("object"):
            cls = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            x = xmin / width
            y = ymin / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            region_id = obj.find("id").text if obj.find("id") is not None else "unknown"
            color = obj.find("color").text if obj.find("color") is not None else "#000000"
            highlighted = obj.find("highlighted").text.lower() == "true" if obj.find("highlighted") is not None else False

            region = {
                "type": "box",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "highlighted": highlighted,
                "editingLabels": False,
                "color": color,
                "cls": cls,
                "id": region_id
            }
            regions.append(region)

        response_data = {
            "pixelSize": pixelSize,
            "regions": regions,
            "image_name": image_name,
            "project_name": project_name
        }

        return response_data

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing request: {str(e)}"})


# Model_training_API

# Custom exception for training interruption
class TrainingInterrupted(Exception):
    """Custom exception to handle training interruption."""
    pass

# Pydantic model for request validation
class TrainModelRequest(BaseModel):
    project_name: str

# # Signal handler function for interruption
# def handle_interruption(signal, frame):
#     logger.info("Training interrupted")
#     raise TrainingInterrupted

# # Setting up signal handlers to catch interruption signals
# signal.signal(signal.SIGINT, handle_interruption)  # Catch Ctrl+C
# signal.signal(signal.SIGTERM, handle_interruption)  # Catch termination signal
# signal.signal(signal.SIGHUP, handle_interruption)  # Catch terminal closure
# signal.signal(signal.SIGTSTP, handle_interruption) # Catch Ctrl+Z

# Function to generate a unique output directory
def generate_output_dir(base_dir, project_name):
    output_dir = base_dir / project_name
    count = 1
    while output_dir.exists():
        output_dir = base_dir / f"{project_name}{count}"
        count += 1

    try:
        print(f"Attempting to create output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Successfully created output directory: {output_dir}")
        logger.info(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Failed to create output directory: {e}")
        logger.error(f"Failed to create output directory: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create output directory")

    return output_dir


# Function to parse class names from XML annotations
def parse_class_names(annotations_dir):
    class_name_to_id = {}
    current_id = 0
    for xml_file in annotations_dir.iterdir():
        if xml_file.suffix == '.xml':
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_name_to_id:
                    class_name_to_id[class_name] = current_id
                    current_id += 1
    logger.info(f"Class names and IDs: {class_name_to_id}")
    return class_name_to_id

# Function to convert XML annotations to YOLO format text files
def convert_xml_to_txt(annotations_dir, class_name_to_id, output_txt_dir):
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    for xml_file in annotations_dir.iterdir():
        if xml_file.suffix == '.xml':
            tree = ET.parse(xml_file)
            root = tree.getroot()

            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)
            output_file = output_txt_dir / f"{xml_file.stem}.txt"
            with output_file.open('w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_id = class_name_to_id[class_name]
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    x_center = (xmin + xmax) / 2.0 / img_width
                    y_center = (ymin + ymax) / 2.0 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            logger.info(f"Converted {xml_file} to {output_file}")

# Function to organize dataset into train and validation sets
def organize_dataset(images_dir, annotations_dir, output_dir):
    train_images_dir = output_dir / 'images/train'
    val_images_dir = output_dir / 'images/val'
    train_labels_dir = output_dir / 'labels/train'
    val_labels_dir = output_dir / 'labels/val'

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    images = [img for img in images_dir.iterdir() if img.suffix == '.jpg']
    valid_images = [img for img in images if (annotations_dir / f"{img.stem}.txt").exists()]
    train_split = int(0.8 * len(valid_images))  # 80% training, 20% validation

    train_images = valid_images[:train_split]
    val_images = valid_images[train_split:]

    for img in train_images:
        shutil.copy(img, train_images_dir)
        annotation_file = annotations_dir / f"{img.stem}.txt"
        shutil.copy(annotation_file, train_labels_dir)

    for img in val_images:
        shutil.copy(img, val_images_dir)
        annotation_file = annotations_dir / f"{img.stem}.txt"
        shutil.copy(annotation_file, val_labels_dir)

    logger.info(f"Dataset organized. Train images: {len(train_images)}, Validation images: {len(val_images)}")

# Function to create YAML file specifying dataset details
def create_yaml_file(output_dir, class_name_to_id, yaml_path='data.yaml'):
    data = {
        'train': str(output_dir / 'images/train'),
        'val': str(output_dir / 'images/val'),
        'nc': len(class_name_to_id),
        'names': list(class_name_to_id.keys())
    }
    yaml_full_path = output_dir / yaml_path
    with yaml_full_path.open('w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    logger.info(f"YAML file created at: {yaml_full_path}")
    return yaml_full_path


from fastapi import BackgroundTasks
@router.post("/train_model/")
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    async def train():
        try:
            project_name = request.project_name
            print(f"🔥 Training function started for project: {project_name}")
            images_dir = BASE_DIR / project_name
            annotations_dir = images_dir
            base_dir = TRAINING_DATA_DIR

            if not images_dir.exists() or not images_dir.is_dir():
                raise HTTPException(status_code=404, detail=f"Images directory does not exist for project: {project_name}")

            start_time = datetime.now()
            training_doc = {"project_name": project_name, "start_time": start_time}

            # Organize dataset
            output_dir = generate_output_dir(base_dir, project_name)
            output_txt_dir = output_dir / 'txt_dir'
            class_name_to_id = parse_class_names(annotations_dir)
            convert_xml_to_txt(annotations_dir, class_name_to_id, output_txt_dir)
            organize_dataset(images_dir, output_txt_dir, output_dir)
            yaml_path = create_yaml_file(output_dir, class_name_to_id)

            models_dir = MODELS_DIR
            models_dir.mkdir(parents=True, exist_ok=True)

            model = YOLO("yolov8n.yaml")
            epochs=20
            training_doc["epochs"] = epochs
            current_training_collection.insert_one(training_doc)
            # Run training in a separate thread using asyncio.to_thread()
            res = await asyncio.to_thread(model.train, data=yaml_path, epochs=epochs, imgsz=640, project=models_dir, name=project_name)
            
            # latest model .csv path
            results_csv_path = models_dir / project_name / 'results.csv'

            # latest model .pt path
            best_pt_path = Path(res.save_dir) / "weights" / "best.pt"
            if best_pt_path.exists():
                training_doc["best_model_path"] = str(best_pt_path)
            else:
                training_doc["best_model_path"] = None  # Indicate best.pt wasn't found

            if best_pt_path.exists():
                training_doc["best_model_path"] = str(best_pt_path)

                # Copy YAML file to the same directory as best.pt
                yaml_dest_path = best_pt_path.parent / yaml_path.name
                shutil.copy(yaml_path, yaml_dest_path)
                training_doc["yaml_path"] = str(yaml_dest_path)  # Save YAML path in training doc

            else:
                training_doc["best_model_path"] = None
            # Read the mAP value from the last line
            mAP_value = None
            with results_csv_path.open('r') as file:
                last_line = file.readlines()[-1]
                mAP_value = last_line.strip().split(',')[-1]

            # Insert into the current training collection (track ongoing training)
            # current_training_collection.insert_one(training_doc)
            # Move training document from current to completed collection
            current_training_collection.delete_one({"project_name": project_name, "start_time": start_time})
            training_doc["end_time"] = datetime.now()
            float_value = float(mAP_value)  #convert string to float
            mAP_value = round(float_value,2)
            training_doc["mAP"] = mAP_value
            # training_doc["best_model_path"]: best_pt_path
            
            trained_models_collection.insert_one(training_doc)

            print("🔥 Training completed successfully!")

        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")

    # Run training in the background
    background_tasks.add_task(train)
    
    return {"status": "in_progress", "detail": "Training started in the background."}


# API to Get all trained model 

class TrainedModelResponse(BaseModel):
    id: str
    project_name: str
    start_time: str
    end_time: Optional[str]
    mAP: Optional[str]

# Helper function to convert MongoDB document to Pydantic model
def document_to_trained_model_response(doc):
    return TrainedModelResponse(
        id=str(doc["_id"]),
        project_name=doc["project_name"],
        start_time=doc["start_time"].isoformat(),
        end_time=doc.get("end_time").isoformat() if doc.get("end_time") else None,
        mAP=doc.get("mAP")
    )



# [ updated on july 10.]
@router.get("/get_all_trained_models", response_model=List[TrainedModelResponse])
async def get_all_trained_models():
    try:
        # Query all documents from trained_models_collection
        cursor = trained_models_collection.find({})
        documents = await cursor.to_list(length=None)  # Fetch all documents

        # Organize documents by project name
        project_name_map = {}
        for doc in documents:
            project_name = doc["project_name"]
            if project_name not in project_name_map:
                project_name_map[project_name] = []
            project_name_map[project_name].append(doc)

        # Rename duplicates based on timestamp
        renamed_documents = []
        for project_name, docs in project_name_map.items():
            if len(docs) > 1:
                # Sort documents by start_time
                docs.sort(key=lambda x: x["start_time"])
                for i, doc in enumerate(docs):
                    if i == 0:
                        renamed_documents.append(doc)
                    else:
                        new_project_name = f"{project_name}_v{i+1}"
                        doc["project_name"] = new_project_name
                        renamed_documents.append(doc)
            else:
                renamed_documents.extend(docs)

        # Convert documents to Pydantic models
        trained_models = [document_to_trained_model_response(doc) for doc in renamed_documents]

        return JSONResponse(content=[model.dict() for model in trained_models])
    except Exception as e:
        logger.error(f"Exception occurred while fetching trained models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Helper function to convert MongoDB document to Pydantic model
def document_to_trained_model_response(doc):
    return TrainedModelResponse(
        id=str(doc["_id"]),
        project_name=doc["project_name"],
        start_time=doc["start_time"].isoformat(),
        end_time=doc.get("end_time").isoformat() if doc.get("end_time") else None,
        mAP=doc.get("mAP")
    )


# read results from model CSV file
class CSVInput(BaseModel):
    project_name: str
    periodic: bool = False  # Flag to start periodic reading

def read_results(file_path):
    if not file_path.exists():
        print("File does not exist:", file_path)
        return {"epoch": None, "metrics/mAP50-95(B)": None}

    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            if not rows:
                print("No rows found in the CSV file.")
                return {"epoch": None, "metrics/mAP50-95(B)": None}

            # Print all headers and rows for debugging
            print("CSV Headers:", reader.fieldnames)
            print("Rows found in the CSV file:")
            for row in rows:
                print(row)

            last_row = rows[-1]
            # Strip spaces from keys
            last_row = {k.strip(): v.strip() for k, v in last_row.items()}
            print("Processed last row:", last_row)
            
            # Explicitly check for headers
            if 'epoch' not in last_row:
                print("'epoch' key not found in CSV headers.")
            if 'metrics/mAP50-95(B)' not in last_row:
                print("'metrics/mAP50-95(B)' key not found in CSV headers.")

            epoch = last_row.get('epoch')
            mAP = last_row.get('metrics/mAP50-95(B)')
            print(f"Extracted values - Epoch: {epoch}, mAP: {mAP}")
            return {"epoch": epoch, "metrics/mAP50-95(B)": mAP}
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {"epoch": None, "metrics/mAP50-95(B)": None}

def copy_and_read_results(latest_csv_path):
    temp_file = TEMP_DIR / "results.csv"
    
    if latest_csv_path.exists():
        try:
            print(f"Copying CSV file from {latest_csv_path} to {temp_file}")
            shutil.copy(latest_csv_path, temp_file)
            results = read_results(temp_file)
            os.remove(temp_file)
            return results
        except Exception as e:
            print(f"Error during file copy or read: {e}")
            return {"epoch": None, "metrics/mAP50-95(B)": None}
    else:
        print(f"CSV file does not exist at path: {latest_csv_path}")
        # List the contents of the directory for debugging
        print(f"Contents of the directory {latest_csv_path.parent}: {os.listdir(latest_csv_path.parent)}")
        return {"epoch": None, "metrics/mAP50-95(B)": None}

def get_latest_csv_path(csv_dir, project_name):
    subdirs = [d for d in Path(csv_dir).iterdir() if d.is_dir() and d.name.startswith(project_name)]
    if not subdirs:
        print(f"No subdirectories found for project name: {project_name}")
        return None

    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    csv_file = latest_subdir / "results.csv"
    print(f"Latest CSV file path: {csv_file}")  # Debug print
    return csv_file


def get_latest_project():
    """Fetch the latest trained project directory"""
    projects = sorted(MODELS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    return projects[0] if projects else None


@router.get("/get-results")
async def get_results():
    
    ongoing_training = current_training_collection.find_one(
            {},  # Empty filter to match any document
            sort=[("start_time", -1)],  # Sort by 'start_time' in descending order
            projection={"_id": 0, "epochs": 1}  # Only return the 'epochs' field
        )
    total_epochs = ongoing_training.get("epochs",0)
    # Check if ongoing_training is None
    if ongoing_training is None:
        return {"No ongoing training"}

    latest_project = get_latest_project()
    
    if not latest_project:
        return {"No ongoing training"}
    
    results_csv_path = latest_project / 'results.csv'
    if results_csv_path.exists():
        df = pd.read_csv(results_csv_path)
        last_epoch = df.iloc[-1]  # Get last row
        epoch_difference = total_epochs-last_epoch
        progress_percentage = (epoch_difference / total_epochs) * 100

    if not results_csv_path.exists():
        return {"No trainning under process"}
    if progress_percentage[0] == 0:
         return {"Progeress %":(100-progress_percentage[0]),"status": "completed"}
    
    return {"Progeress %":(100-progress_percentage[0]),"status": "in_progress"}

# Get Cams Config 

# Function to establish MongoDB connection
async def connect_to_mongodb():
    client = MongoClient(MONGO_URL)
    db = client[DATABASE]
    collection = db[COLLECTION]
    return collection

# GET endpoint to retrieve cams_config.json from MongoDB
@router.get("/get_cams_config", response_model=dict)
async def get_cams_config():
    try:
        collection = await connect_to_mongodb()
        # Retrieve the document containing cams_config.json
        document = await collection.find_one({"document_key": "cams_config"})
        
        if document:
            cams_config = document.get("cams_config", {})
            return cams_config
        else:
            raise HTTPException(status_code=404, detail="Cams config not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


