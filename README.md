# NYU UltraViolet Midterm 2 practical
~ Shubham Gupta

## Installation
### Prompt 1,2
- install Anaconda
- install CUDA
```shell
conda env create -f environment.yml
```
### Prompt 3
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
```shell
systemctl start cv_detection
systemctl enable cv_detection
```

### References
- https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
- https://developer.nvidia.com/cuda-downloads
- https://github.com/NYU-Robomaster-Ultraviolet/CV_Detection
- https://medium.com/@benmorel/creating-a-linux-service-with-systemd-611b5c8b91d6
