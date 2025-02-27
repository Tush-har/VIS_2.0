import os
import shutil
import signal
import csv
import time
import zipfile
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
print(".........test......")
import ultralytics
from ultralytics import YOLO
from fastapi.responses import JSONResponse
# from pathlib import BaseModel
from fastapi import FastAPI, Request
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
from fastapi.responses import JSONResponse
from bson import ObjectId
from typing import List, Optional



# Load environment variables from a .env file
load_dotenv()

# Constants and paths
BASE_DIR = Path(os.getenv("BASE_DIR", "C:/SPRL/Project"))
TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", "C:/SPRL/Training_data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "C:/SPRL/Models"))
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

# Hardcoded base directory for API to read results from CSV (read_results)
CSV_DIR = "C:/SPRL/Models"
TEMP_DIR = Path("C:/SPRL/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI application instance
app = FastAPI()

# MongoDB client
client = AsyncIOMotorClient(MONGODB_URL)
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



@app.post("/upload_data/")
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



@app.get("/get_all_project/")
async def get_projects():
    try:
        project_dirs = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]
        return {"projects": project_dirs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})




class Project(BaseModel):
    project_name: str

@app.post("/create_project/") 
async def create_project(request: Request, project: Project):
    try:
        project_name = project.project_name
        project_dir = BASE_DIR / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        return {"message": f"Project '{project_name}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_project_data/")
async def get_project_data(request: Request,project: Project):
    try:
        project_name = project.project_name
        project_dir = BASE_DIR / project_name
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

        # Get all image and text file names in the project directory
        image_files = [f.name for f in project_dir.glob("*.jpg")]
        txt_files = [f.name for f in project_dir.glob("*.txt")]

        return {
            "images": image_files,
            "text_files": txt_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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

@app.post("/upload_data/save_annotation")
async def save_annotation(payload: Payload):
    try:
        xml_annotation = create_xml_annotation(payload)
        save_annotation_to_file(xml_annotation, f"{payload.project_name}/{payload.image_name}.xml")
        return {"xml": xml_annotation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {str(e)}")


# get annotations API

class AnnotationRequest(BaseModel):
    project_name: str
    image_name: str

@app.post("/get_annotation/")
async def get_annotation(request: AnnotationRequest):
    try:
        project_name = request.project_name
        image_name = request.image_name
        annotation_path = BASE_DIR / project_name / f"{image_name}.xml"

        # Log the constructed annotation path
        print(f"Looking for annotation file at: {annotation_path}")

        if not annotation_path.exists():
            error_message = f"File not found for project: {project_name}, image: {image_name}"
            print(error_message)
            response_json = {
                "message": error_message,
                "project_name": project_name,
                "image_name": image_name
            }
            return response_json
        
        # Parse the XML file
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Extract image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        pixelSize = {"w": width, "h": height}

        regions = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Normalize bounding box coordinates
            x = xmin / width
            y = ymin / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # Get additional attributes
            region_id = obj.find('id').text
            color = obj.find('color').text
            highlighted = obj.find('highlighted').text.lower() == 'true'

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
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")


# Model_training_API

# Custom exception for training interruption
class TrainingInterrupted(Exception):
    """Custom exception to handle training interruption."""
    pass

# Pydantic model for request validation
class TrainModelRequest(BaseModel):
    project_name: str

# Function to generate a unique output directory
def generate_output_dir(base_dir, project_name):
    output_dir = base_dir / project_name
    count = 1
    while output_dir.exists():
        output_dir = base_dir / f"{project_name}{count}"
        count += 1
    output_dir.mkdir(parents=True)
    logger.info(f"Created output directory: {output_dir}")
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

# FastAPI endpoint to train the model
@app.post("/train_model/")
async def train_model(request: TrainModelRequest):
    try:
        project_name = request.project_name
        images_dir = BASE_DIR / project_name
        annotations_dir = images_dir
        base_dir = TRAINING_DATA_DIR

        # Ensure images and annotations directories exist
        if not images_dir.exists() or not images_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Images directory does not exist for project: {project_name}")
        
        # Log training start time and project details in current training collection
        start_time = datetime.utcnow()
        training_doc = {"project_name": project_name, "start_time": start_time}
        await current_training_collection.insert_one(training_doc)

        # Organize the dataset
        output_dir = generate_output_dir(base_dir, project_name)
        output_txt_dir = output_dir / 'txt_dir'

        class_name_to_id = parse_class_names(annotations_dir)
        convert_xml_to_txt(annotations_dir, class_name_to_id, output_txt_dir)
        organize_dataset(images_dir, output_txt_dir, output_dir)
        yaml_path = create_yaml_file(output_dir, class_name_to_id)

        # Define the path for the models directory
        models_dir = MODELS_DIR

        # Create the models directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)

        # Train the model and specify the project path and name for the saved model
        model = YOLO("yolov8s.yaml")
        model.train(data=yaml_path, epochs=40, imgsz=640, project=models_dir, name=project_name, device=0)

        # Get the path to the result.csv file
        results_csv_path = models_dir / project_name / 'results.csv'

        # Read the mAP value from the last line of the result.csv file
        mAP_value = None
        with results_csv_path.open('r') as file:
            last_line = file.readlines()[-1]
            # Assuming mAP value is in the last column of the CSV file
            mAP_value = last_line.strip().split(',')[-1]

        # Move the document from current training to trained models collection
        await current_training_collection.delete_one({"project_name": project_name, "start_time": start_time})
        training_doc["end_time"] = datetime.utcnow()
        training_doc["mAP"] = mAP_value
        await trained_models_collection.insert_one(training_doc)

        return {"status": "completed", "detail": "Training completed successfully!", "mAP": mAP_value}

    except HTTPException as e:
        return {"status": "error", "detail": f"HTTPException: {str(e)}"}
    except TrainingInterrupted:
        return {"status": "error", "detail": "Training interrupted or hampered."}
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        return {"status": "error", "detail": f"Exception occurred: {str(e)}"}
 
# API to Get all trained model 
# Pydantic model for response
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

@app.get("/get_all_trained_models/", response_model=List[TrainedModelResponse])
async def get_all_trained_models():
    try:
        # Query all documents from trained_models_collection
        cursor = trained_models_collection.find({})
        documents = await cursor.to_list(length=None)  # Fetch all documents

        # Convert documents to Pydantic models
        trained_models = [document_to_trained_model_response(doc) for doc in documents]

        return JSONResponse(content=[model.dict() for model in trained_models])
    except Exception as e:
        logger.error(f"Exception occurred while fetching trained models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


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


@app.post("/read_results/")
def get_results(input: CSVInput):
    try:
        if input.periodic:
            # Start a separate thread to read results periodically
            thread = threading.Thread(args=(CSV_DIR, input.project_name), daemon=True)
            thread.start()
            return JSONResponse(content={"status": "success", "detail": "Started periodic reading of results."})
        else:
            latest_csv_path = get_latest_csv_path(CSV_DIR, input.project_name)
            if not latest_csv_path:
                raise HTTPException(status_code=404, detail="No training folders found.")

            results = copy_and_read_results(latest_csv_path)
            return JSONResponse(content=results)
    except Exception as e:
        print(f"Exception: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": f"Error reading results: {str(e)}"})


# Get Cams Config 

# Function to establish MongoDB connection
async def connect_to_mongodb():
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DATABASE]
    collection = db[COLLECTION]
    return collection

# GET endpoint to retrieve cams_config.json from MongoDB
@app.get("/get_cams_config", response_model=dict)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)


