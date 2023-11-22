from ultralytics import YOLO
import tkinter

class Model:
    def __init__(self):
        self.model = None
        self.test_image_paths = None
        self.test_answer_paths = None
        self.last_predicition = None
        
    def make_prediction(self, index):
        if self.model is not None:
            self.last_predicition =  self.model.predict(source = self.test_image_paths[index], conf = 0.35)
            return self.last_predicition
        else:
            return "No model is loaded"