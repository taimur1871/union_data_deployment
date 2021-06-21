# Data Processing Deployment

This repo is to build a basic deployment app for Machine Learning process developed for analysing membership data.

### Environment
* fastapi 0.63.0
* gunicorn 20.0.4
* Jinja2 2.11.2
* pandas 1.2.4
* seaborn 0.11.0
* sklearn 0.23.2
* requests 2.22.0
* uvicorn 0.13.3
* wbgapi 1.0.5

```complete list canbe found in requirements.txt```

## Usage
There are two possible ways to use this app
1. Run Locally
    * To run the app locally create a clone of the app. Then change the port number in app.py to any other than 80 (default port for internet access).
    * The default port is set to 80. If running locally change that to any port of your choice other than 80
    * Install requirements by using "pip install requirements.txt"
    * Start app by typing "python app.py" in terminal or command shell.
2. Run Using Docker
    It is also possible to create a docker container.
    * Just Open the folder in terminal or command shell.
    * Use the command ```docker build -t <app_name> . ```
    * Once the app is built use the command ```docker run --rm --name <app_name_running> -p {local_port}:{docker_image_port} -ti <container_name>```

## Deployment
Before deplyment make sure to change the upload folder to a storage bucket. You can pass the storage bucket address as ENV variable of a Docker container.

The current set up would save any uploaded file to the upload folder in the Docker container which can be accessed manually but that is not a good practice.

(GCP instructions can be found here https://cloud.google.com/run/docs/quickstarts/build-and-deploy/python)