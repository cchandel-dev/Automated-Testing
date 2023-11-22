import unittest
from CV-program import Model, calculate_iou, align_annotations, number_of_correct_class_predictions, number_of_high_iou_bounding_boxes, generate_predictions_and_annotations

class TestStudentFunctions(unittest.TestCase):
    def test_generate_predictions_and_annotations(self):
        my_model = Model()
        for index in range(len(my_model.test_image_paths)):
            with self.subTest(value=index):
                actual_predictions, actual_annotations = generate_predictions_and_annotations(index)
                # write loop to read expected predictions and annotations
                self.assertEqual(len(actual_predictions), len(expected_predictions))
                self.assertEqual(len(actual_annotations), len(expected_annotation))

    def test_number_of_correct_class_predictions(self):
        my_model = Model()
        for index in range(len(my_model.test_image_paths)):
            with self.subTest(value=index):
                actual_predictions, actual_annotations = generate_predictions_and_annotations(index)
                allign = align_annotations(actual_predictions, actual_annotations)
                actual_number_of_ccs  = number_of_correct_class_predictions(actual_predictions, actual_annotations, allign)
                # write loop to read expected number of correct classes
                self.assertEqual(actual_number_of_ccs, expected_number_of_ccs)
    
    def test_number_of_high_iou_bounding_boxes(self):
        my_model = Model()
        for index in range(len(my_model.test_image_paths)):
            with self.subTest(value=index):
                actual_predictions, actual_annotations = generate_predictions_and_annotations(index)
                allign = align_annotations(actual_predictions, actual_annotations)
                actual_number_of_bbs  = number_of_high_iou_bounding_boxes(actual_predictions, actual_annotations, allign)
                # write loop to read expected number of bounding boxes that meet the threshold
                self.assertEqual(actual_number_of_bbs, expected_number_of_bbs)
    
    def test_IOU(self):
        my_model = Model()
        for index in range(len(my_model.test_image_paths)):
            with self.subTest(value=index):
                actual_predictions, actual_annotations = generate_predictions_and_annotations(index)
                allign_by_iou = align_annotations(actual_predictions, actual_annotations)
                # write loop to read expected intersection-over-union
                self.assertAlmostEqual(allign_by_iou[i][2], expected_iou, places=2)


if __name__ == '__main__':
    unittest.main()
