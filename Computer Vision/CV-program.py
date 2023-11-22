import os
from ultralytics import YOLO
from tkinter import filedialog, Listbox, Scrollbar

class Model:
    def __init__(self):
        self.model = YOLO(filedialog.askopenfilename())
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

def align_annotations(predictions, annotations):
    """
    Align predicted bounding boxes with ground truth bounding boxes based on IoU.

    Parameters:
    - predictions: List of predicted bounding boxes in YOLO format.
    - annotations: List of ground truth bounding boxes in YOLO format.
    - threshold: IoU threshold for matching.

    Returns:
    - aligned_pairs: List of tuples (prediction_index, annotation_index) representing aligned pairs.
    """
    aligned_pairs = []

    for pred_index, prediction in enumerate(predictions):
        for annot_index, annotation in enumerate(annotations):
            iou = calculate_iou(prediction[0:4], annotation[0:4]) #only need these four label
            if iou > 0.3:
                aligned_pairs.append((pred_index, annot_index, iou))
    return aligned_pairs


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1, box2: Bounding boxes in YOLO format (x_center, y_center, width, height).

    Returns:
    - IoU: Intersection over Union value.
    """
    # Convert YOLO format to (x_min, y_min, x_max, y_max) format
    box1 = [
        box1[0] - box1[2] / 2,
        box1[1] - box1[3] / 2,
        box1[0] + box1[2] / 2,
        box1[1] + box1[3] / 2
    ]
    box2 = [
        box2[0] - box2[2] / 2,
        box2[1] - box2[3] / 2,
        box2[0] + box2[2] / 2,
        box2[1] + box2[3] / 2
    ]

    # Calculate intersection coordinates
    x_intersection = max(box1[0], box2[0])
    y_intersection = max(box1[1], box2[1])
    w_intersection = max(0, min(box1[2], box2[2]) - x_intersection)
    h_intersection = max(0, min(box1[3], box2[3]) - y_intersection)

    # Calculate area of intersection and union
    area_intersection = w_intersection * h_intersection
    area_union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - area_intersection

    # Calculate IoU
    iou = area_intersection / area_union if area_union > 0 else 0.0

    return iou
