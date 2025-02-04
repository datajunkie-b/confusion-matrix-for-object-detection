from shapely.geometry import Polygon
import os
import cv2
import numpy as np
import csv

def read_txt(txt_file, img_width, img_height, pred=True):
    """
    Parameters
    ----------
    txt_file : str
        Path to the text file to read.
    img_width : int
        Width of the image in pixels.
    img_height : int
        Height of the image in pixels.
    pred : bool, optional
        If True, reads predictions with confidence values. If False, reads ground truth (no confidence). 
        Default is True.
    
    Returns
    -------
    info : list
        A list containing tuples:
        - For predictions (pred=True): (det_class, confd, [(x1, y1), (x2, y2), ...])
        - For ground truth (pred=False): (det_class, [(x1, y1), (x2, y2), ...])
    """
    x = []
    with open(txt_file, 'r') as f:
        info = []
        lines = f.readlines()

        for item in lines:
            item = item.strip().split(" ")

            if pred:
    # For predictions with confidence
                det_class = item[0]
                confd = float(item[-1])
                x1 = float(item[1])
                y1 = float(item[2])
                x2 = float(item[3])
                y2 = float(item[4])
    
    # Define the polygon coordinates in the desired order
                polygon_coords = [
                            (x1, y1), (x2, y1),
                            (x2, y2), (x1, y2)
                                 ]

                # Append the tuple (det_class, confd, polygon_coords)
                info.append((polygon_coords, confd, det_class))

            else:
                # For ground truth (no confidence)
                det_class = item[0]
                coords = item[1:]
                
                # Normalize and convert to absolute coordinates
                polygon_coords = [
                    (
                        int(float(coords[i]) * img_width), 
                        int(float(coords[i+1]) * img_height)
                    )
                    for i in range(0, len(coords), 2)
                ]

                # Append the tuple (det_class, polygon_coords)
                info.append((det_class, polygon_coords))

    return info

def process_files(txt_directory, img_directory, pred=True):
    """
    Processes text files and their corresponding image files.
    
    Parameters
    ----------
    txt_directory : str
        Directory containing text files.
    img_directory : str
        Directory containing image files.
    pred : bool, optional
        If True, reads predictions with confidence values. If False, reads ground truth. 
        Default is True.
    
    Returns
    -------
    all_boxes_with_names : list
        A list of tuples containing text file names and the corresponding bounding box data.
    """
    # List of valid image file extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', 'JPG', 'JPEG', 'PNG')

    all_boxes_with_names = []

    for txt_file in os.listdir(txt_directory):
        if txt_file.endswith(".txt"):
            base_name = os.path.splitext(txt_file)[0]

            # Find the corresponding image file with any valid extension
            img_file_path = None
            for ext in valid_extensions:
                potential_img_path = os.path.join(img_directory, base_name + ext)
                if os.path.exists(potential_img_path):
                    img_file_path = potential_img_path
                    break

            if img_file_path:
                # Read image and get dimensions
                img = cv2.imread(img_file_path)
                img_height, img_width, _ = img.shape

                # Process the corresponding text file
                txt_file_path = os.path.join(txt_directory, txt_file)
                boxes = read_txt(txt_file_path, img_width, img_height, pred)
            else:
                # No corresponding image file found
                boxes = []

            # Append file name along with boxes (empty or not)
            all_boxes_with_names.append((txt_file, boxes))

    return all_boxes_with_names

# Directories for images
txt_directory_pred = r'C:\Users\vishal.ponuganti\Downloads\Polygons\Predictions\detections_txt\detections_txt'
img_directory_pred = r'C:\Users\vishal.ponuganti\Downloads\Polygons\Predictions\detections_images\detections_images'
pred_polygon = process_files(txt_directory_pred,img_directory_pred)


# Directories for labels
txt_directory_gt = r'C:\Users\vishal.ponuganti\Downloads\project-7-at-2025-01-20-10-21-7fd8cfcf\labels\label_renamed_directory'
img_directory_gt = r'C:\Users\vishal.ponuganti\Downloads\project-7-at-2025-01-20-10-21-7fd8cfcf\images\image_renamed_directory'
gt_polygon = process_files(txt_directory_gt,img_directory_gt,pred=False)


class_names = ['Lid','Manhole_Closed']


class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.3):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))  # Including an extra row/column for false positives and negatives
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
        self.TN_count = 0

    def compute_iou(self, polygon1, polygon2):
        """
        Compute the IoU (Intersection over Union) between two polygons using Shapely.
        """
        poly1 = Polygon(polygon1)
        poly2 = Polygon(polygon2)

        # Ensure valid polygons
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area 

        return intersection / union if union > 0 else 0

    def update(self, pred_boxes_with_names, gt_boxes_with_names):
        """
        Update the confusion matrix with predicted and ground truth boxes (polygons).
        """
        for pred_file, pred_boxes in pred_boxes_with_names:
            # Retrieve the ground truth boxes for the corresponding file
            gt_boxes = next((g[1] for g in gt_boxes_with_names if g[0] == pred_file), [])

            gt_matched = set()
            pred_matched = set()

            if pred_boxes == [] and gt_boxes == []: 
                self.TN_count += 1
                with open(output_csv, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([pred_file, 'N/A', 'N/A', 'TN'])
                continue

            # Iterate through the predicted polygons
            for pred_idx, pred in enumerate(pred_boxes):
                pred_coords, conf, pred_class = pred
                if conf < self.CONF_THRESHOLD:
                    continue

                matched = False
                for gt_idx, gt in enumerate(gt_boxes):
                    gt_class,gt_coords = gt

                    if gt_idx in gt_matched:
                        continue

                    # Calculate IoU between predicted and ground truth polygons
                
                    iou = self.compute_iou(pred_coords, gt_coords)
                    if iou >= self.IOU_THRESHOLD:
                        if pred_class == gt_class:
                            self.matrix[int(gt_class), int(pred_class)] += 1
                            gt_matched.add(gt_idx)
                            pred_matched.add(pred_idx)
                            matched = True

                            with open(output_csv, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([pred_file, 
                                                 f"{class_names[int(gt_class)]},({','.join(map(str, gt_coords))})",
                                                 f"{class_names[int(pred_class)]},{','.join(map(str, pred_coords))}",
                                                 'TP'])
                            break
                        else:
                            self.matrix[int(gt_class), int(pred_class)] += 1
                            gt_matched.add(gt_idx)
                            pred_matched.add(pred_idx)
                            matched = True
    

                            with open(output_csv, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([pred_file, 
                                                 f"{class_names[int(gt_class)]},({','.join(map(str, gt_coords))})",
                                                 f"{class_names[int(pred_class)]},{','.join(map(str, pred_coords))}",
                                                 'FN'])

                if not matched:
                    # False positive
                    self.matrix[self.num_classes, int(pred_class)] += 1

                    with open(output_csv, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([pred_file, 'N/A',
                                         f"{class_names[int(pred_class)]},({','.join(map(str, pred_coords))})",
                                         'FP'])

            # Count unmatched ground truth polygons as false negatives
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx not in gt_matched:
                    gt_class, gt_coords = gt
                    self.matrix[int(gt_class), self.num_classes] += 1

                    with open(output_csv, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([pred_file,
                                         f"{class_names[int(gt_class)]},({','.join(map(str, gt_coords))})",
                                         'N/A', 'FN'])
                        


    def calculate_metrics(self):
        """
        Calculate precision, recall, and accuracy for each class.
        """
        precisions = []
        recalls = []
        accuracies = []
        TP_list = []
        FP_list = []
        FN_list = []

        for cls in range(self.num_classes):
            TP = self.matrix[cls, cls]
            FP = self.matrix[self.num_classes, cls]
            FN = np.sum(self.matrix[cls, :]) - TP
            TN = self.TN_count/len(class_names)

            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            accuracy = (TP + TN) / (TP+TN+FP+FN) if TP+TN+FP+FN > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            accuracies.append(accuracy)

            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)


        return precisions, recalls, accuracies
                        
class_names = ['Lid','Manhole_Closed']

conf_matrix = ConfusionMatrix(num_classes=len(class_names))

output_csv = r"C:\Users\vishal.ponuganti\Downloads\project-7-at-2025-01-20-10-21-7fd8cfcf\poly_detections.csv"
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Groundtruth', 'Prediction', 'TP/FN/FP'])

conf_matrix.update(pred_polygon, gt_polygon)

print("Confusion Matrix:\n", conf_matrix.matrix)

precisions, recalls, accuracies = conf_matrix.calculate_metrics()
print("\nClass-wise Metrics:")
for i, class_name in enumerate(class_names):
    print(f"Class: {class_name}")
    print(f"  Precision: {precisions[i]:.2f}")
    print(f"  Recall: {recalls[i]:.2f}")
    print(f"  Accuracy: {accuracies[i]:.2f}")
