# NYU UltraViolet Midterm 2 practical
~ Shubham Gupta

## Submission questions
### Design Decisions
#### Prompt1 
- Develop a class based solution. This allows code to be modularized and can be reused / refactored with other future projects.
- Used inbuilt python logger module. This sets the logging level for each python file and again allows to debug classes independently.
- The plates are first detected using the CV torch model. The plate coordinates are then passed on for preprocessing where I detected the colour. If the plates HSV range matches the blue colour HSV range, we draw bounding boxes. This is much faster than first finding blue colour and then running multiple instances of inference on the the image.
### Prompt 2
- Developed prompt 2 as part of prompt 1. The offset is calculated as a function of the ProcessFrame class. This made sense logically as we only need to get offset once a blue plate is detected (in process_frame method).
### Prompt 3:
- The SystemD service shown only has the structure the service and is not actually designed to run. 
- The service type is simple. The remaining types do not fit the requirements of the service. 'exec' was another option but it will not proceed before both fork() and execve() in the service process.
### Prompt 4:
<p align="center">
    <img src="./doc/multi_plate.png">
</p>

## Installation
### Prompt 1,2
- install Anaconda
- install CUDA
```shell
# use yml file based on OS
conda env create -f conda_envs/environment_win.yml
```
### Prompt 3
<b>Prompt 3 only shows the structure of the systemD file and how it should be stored.</b>
```shell
cp src/systemd/cv_detection.service /etc/systemd/system/cv_detection.service
```

## Run
### Prompt 1,2
<b>Make sure to have atleast 1 camera connected / add video path</b> 

From the base repository
```shell
cd src
python main.py
```

### Prompt 3
<b>Edit path to main file before enabling the service</b>
```shell
systemctl start cv_detection
systemctl enable cv_detection
```

### Update yml files
#### Linux or Mac
```shell
conda env export --no-builds | grep -v "prefix" > environment.yml
```
#### Windows
```shell
conda env export --no-builds | findstr -v "prefix" > environment.yml
```

### References
- https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
- https://developer.nvidia.com/cuda-downloads
- https://github.com/NYU-Robomaster-Ultraviolet/CV_Detection
- https://medium.com/@benmorel/creating-a-linux-service-with-systemd-611b5c8b91d6
