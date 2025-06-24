from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Allow all origins during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origin for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve your Zarr dataset folder
app.mount("/data", StaticFiles(directory="public/data"), name="data")
