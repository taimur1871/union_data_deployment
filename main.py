#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
created on Tue Jun 15
'''

# Library imports
from typing import List
from fastapi.datastructures import UploadFile

from wbgapi.source import features
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# import chart and stats
from utils.save_upload import save_uploaded_file
from utils.read_excel import parse_contents

# python modules
import time
import os, shutil
from pathlib import Path

# data processing
import pandas as pd

# Create app and model objects
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/upload", StaticFiles(directory="upload"), name="upload")
templates = Jinja2Templates(directory="templates/")

# Welcome page
@app.get("/", response_class=HTMLResponse)
async def read_root(request:Request):
    # get a list of available features
    return templates.TemplateResponse("welcome.html", {"request": request})

# data properties page
@app.post("/uploadfile")
async def read_root(request:Request, files: List[UploadFile] = File(...)):
    
    # create an upload folder directory using timestamp
    upload_folder = os.path.join('./upload', time.ctime(time.time()))
    os.makedirs(upload_folder)
    
    # upload files, kept multi file upload option for now
    for file in files:
        fn = file.filename

        p = Path(upload_folder +'/'+ fn)

        # exception handling for no files uploaded
        try:
            save_uploaded_file(file, p)
        except IsADirectoryError:
            message = 'no files uploaded, please try again'
            return templates.TemplateResponse("error_page.html", 
            {"request":request, 
            "message":message})

    # open file and get dataframe
    df, df_head = parse_contents(p, fn)

    return templates.TemplateResponse("upload.html", 
    {"request": request, "dataset_name":p,
    "df_head":[df_head.to_html(classes='data', header='true')]})