import os
from ultralytics import YOLO
from tkinter import filedialog, Listbox, Scrollbar

class Model:
    def __init__(self):
        self.model = filedialog.askopenfilename()
        self.test_image_paths =  [f for f in os.list_dir(filedialog.askdirectory()) if f.endswith('.png')]
        self.test_answer_paths = [f for f in os.list_dir(filedialog.askdirectory()) if f.endswith('.txt')]
        self.last_predicition = None
        
    def make_prediction(self, index):
        if self.model is not None:
            self.last_predicition =  self.model.predict(source = self.test_image_paths[index], conf = 0.35)
            return self.last_predicition
        else:
            return "No model is loaded"
    
    def get_last_prediction_bounding_boxes(self):
        return len(self.last_prediction.__getitem__(0).boxes.xywhn)
    
    def get_last_prediction_classes(self):
        return len(self.last_prediction.__getitem__(0).boxes.cls)
    