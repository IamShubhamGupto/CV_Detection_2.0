import torch
from pathlib import Path
import logging

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(level=logging.DEBUG)

class TorchModel:
    def __init__(self,
        repo_name='ultralytics/yolov5',
        weights_path='./pt_files/best.pt',
        model_name = 'custom'
        ):
        self.weights_path = weights_path
        self.repo_name = repo_name
        self.model_name = model_name
        logger.debug('initialized TorchModel class')
    
    # Model
    def get_model(self):
        try:
            if torch.backends.mps.is_available():
                mps_device = torch.device("mps")
                model = torch.hub.load(self.repo_name, self.model_name, path=self.weights_path, device=mps_device)  # or yolov5m, yolov5l, yolov5x, custom
                logger.debug('torch model loaded with MPS')
                return model
        except:
            logger.debug('running non mac OS ')
        model = torch.hub.load(self.repo_name, self.model_name, path=self.weights_path)  # or yolov5m, yolov5l, yolov5x, custom
        logger.debug('torch model loaded')
        return model

    # Returns coordinates
    def get_coordinates(self, frame, model):
        results = model(frame)                  # using the model each frame
        rows = results.pandas().xyxy[0]  
        if len(rows) != 0:
            x_min, y_min, x_max,y_max = int(rows['xmin'][0]), int(rows['ymin'][0]), int(rows['xmax'][0]), int(rows['ymax'][0]) 
            logger.debug('got coordinates = %d %d %d %d'.format(x_min, y_min, x_max, y_max))
            return (x_min, y_min, x_max, y_max)
        logger.debug('got no coordinates')
        return None