#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Library imports
from typing import List

from wbgapi.source import features
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# import chart and stats
from utils import api_new
import pandas as pd

# Create app and model objects
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/")

# define list of features at application start up
# setting this globally for testing only
features, feat_codes = api_new.options()

# Welcome page
@app.get("/", response_class=HTMLResponse)
async def read_root(request:Request):
    # get a list of available features
    return templates.TemplateResponse("welcome.html", {"request": request,
    'features':features})

# data properties page
@app.get("/getdata", response_class=HTMLResponse)
async def read_root(request:Request):
    # retrieve category value
    req_params = request.query_params
    keyword = req_params.multi_items()

    # get feature codes using utils function
    feat_code = api_new.get_code(keyword[0][1], feat_codes)

    # get the properties of the dataset
    prop = api_new.request_data(feat_code)

    return templates.TemplateResponse("welcome.html", 
    {"request": request, "features":features,
    "dataset_name":keyword[0][1],
    "prop":[prop.to_html(classes='data', header='true')]})

''' 
function originally meant to show results on a separate page,
not being used here for now

# get the data and process it to a pandas dataframe 
# can be changed to dask or other formats
@app.post("/load_data")
async def create_upload_files(request:Request, keyword):
    
    # send api call to get the required data
    data_obj = api_new.laod_data(keyword)
    
    # create initial dataframe
    df_ini = pd.DataFrame(data_obj)

    return templates.TemplateResponse("report.html", {"request":request, "message":message,
                                    "inner":inner, "outer":outer, "main":main, "second":second,
                                    "loc":loc, "top_name": top_name_disp, "blade_pic":blade_1,
                                    "SN":temp_folder[0]})
'''