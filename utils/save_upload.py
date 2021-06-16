# python3.8

# module imports
from fastapi import UploadFile
import shutil
from pathlib import Path

# function to save uploaded files
def save_uploaded_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()