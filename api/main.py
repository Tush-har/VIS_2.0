
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from user.endpoints import router as user_router
from annotation.endpoints import router as annotation_router
from inspections.endpoints import router as inspection_router
from camera_config.endpoints import router as config_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router, prefix='/accounts')
app.include_router(annotation_router, prefix='/annotation')
app.include_router(inspection_router, prefix='/inspection')
app.include_router(config_router,prefix='/config')



if __name__ == "__main__":
    # uvicorn.run(app, host="192.168.10.70", port=8080)
    uvicorn.run(app)