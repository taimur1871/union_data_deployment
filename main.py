#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
created on Tue Jun 15
'''

# Library imports
from typing import List
from fastapi.datastructures import UploadFile

from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# import chart and stats
from utils.save_upload import save_uploaded_file
from utils.read_excel import parse_contents
from process_data.eda import mlp_test, get_preds

# python modules
import time
import os
from pathlib import Path

# data processing
import pandas as pd

# Create app and model objects
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/upload", StaticFiles(directory="upload"), name="upload")
templates = Jinja2Templates(directory="templates/")

# set model upload folder
model_upload_folder = None

# Welcome page
@app.get("/", response_class=HTMLResponse)
async def read_root(request:Request):
    # get a list of available features
    return templates.TemplateResponse("welcome_alt.html",
    {"request": request})

# data properties page
@app.post("/uploadfile")
async def upload(request:Request, background_tasks:BackgroundTasks,
                files: List[UploadFile] = File(...)):
    
    # create an upload folder directory using timestamp
    upload_folder = os.path.join('./upload', time.ctime(time.time()))
    os.makedirs(upload_folder)

    model_path = './models/saved_model'
    
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
    df_pred, _ = parse_contents(p, fn)

    # get predicted dataframe
    df_predicted = get_preds(df_pred, model_path)
    df_predicted.drop('category', axis=1, inplace=True)

    return templates.TemplateResponse("datatable_version.html", 
    {"request": request, 
    "data_summary": [df_predicted.to_html(table_id='table_id').replace('border="1"', ' ')]})