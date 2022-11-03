# NYU UltraViolet Midterm 2 practical
~ Shubham Gupta

## Installation
### Prompt 1,2
- install Anaconda
- install CUDA
```shell
# use yml file based on OS
conda env create -f environment_win.yml
```
### Prompt 3
<b>Prompt 3 only shows the structure of the systemD file and how it should be stored.</b>
```shell
cp src/systemd/cv_detection.service /etc/systemd/system/cv_detection.service
```

## Run
### Prompt 1,2
<b>Make sure to have atleast 1 camera connected</b> 

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
