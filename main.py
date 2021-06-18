#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
created on Tue Jun 15
'''

# Library imports
from typing import List
from fastapi.datastructures import UploadFile
from fastapi.params import ParamTypes

from wbgapi.source import features
from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# import chart and stats
from utils.save_upload import save_uploaded_file
from process_data.eda import pre_process
from utils.read_excel import parse_contents
from process_data.eda import mlp_test

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

# set model upload folder
model_upload_folder = None

# Welcome page
@app.get("/", response_class=HTMLResponse)
async def read_root(request:Request):
    # get a list of available features
    return templates.TemplateResponse("welcome.html", {"request": request})

# data properties page
@app.post("/uploadfile")
async def upload(request:Request, background_tasks:BackgroundTasks,
                files: List[UploadFile] = File(...)):
    
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
    df_train, _ = parse_contents(p, fn)
    df_pred, _ = parse_contents(p, fn)

    df_pred_1 = mlp_test(df_train, df_pred, upload_folder)
    #df_head = df_pred_1.head()

    # create a dataframe for displaying data types
    #df_dtypes = pd.DataFrame(df_train.dtypes)
    #df_dtypes.rename({0:'Data Type'}, axis=1, inplace=True)

    return templates.TemplateResponse("predictions.html", 
    {"request": request, "dataset_name":p, "model_path":upload_folder,
    "data_summary": [df_pred_1.to_html(classes='data', header='true')]})

'''
# process the dataframe
@app.post("/predict")
async def predict_page(request:Request, files: List[UploadFile] = File(...)):
    # get data from query
    req_params = request.query_params
    model_path = req_params.keys()

    return templates.TemplateResponse("predictions.html", 
    {"request":request, "model_path":model_path})

# process the dataframe
@app.get("/predict")
async def predict_page(request:Request):
    # get data from query
    req_params = request.query_params
    model_path = req_params.keys()

    return templates.TemplateResponse("predictions.html", 
    {"request":request, "model_path":model_path})
'''