import numpy as np
import os, copy
import cv2
import seaborn as sns
import csv
from datetime import datetime

def read_txt(txt_file, img_width, img_height, pred=True):
    '''
    Parameters
    ----------
    txt_file : txt file path to read
    pred : if you are reading prediction txt file, it'll have 6 values 
    (i.e., including confidence) whereas GT won't have confd value. So set it
    to False for GT file. The default is True.
    Returns
    -------
    info : a list having 
        if pred=True => detected_class, confd, x_min, y_min, x_max, y_max
        if pred=False => detected_class, x_min, y_min, x_max, y_max
    '''
    x = []
    with open(txt_file, 'r') as f:
        info = []
        x = x + f.readlines()
        for item in x:
            item = item.replace("\n", "").split(" ")
            if pred:
                # for preds because 2nd value in preds is confidence
                det_class = item[0]
                confd = float(item[5])
                x1 = float(item[1])
                y1 = float(item[2])
                x2 = float(item[3])
                y2 = float(item[4])

                info.append((x1, y1, x2, y2, confd, det_class))
                             
            else:
                # for gt 
                det_class = item[0]
                x_c = float(item[1])
                y_c = float(item[2])
                width = float(item[3])
                height = float(item[4])

                # absolute coordinates
                x_c_abs = x_c * img_width
                y_c_abs = y_c * img_height
                width_abs = width * img_width
                height_abs = height * img_height  

                # Calculate the x1, y1, x2, y2 coordinates
                x1 = int((x_c_abs - width_abs / 2))
                y1 = int((y_c_abs - height_abs / 2))
                x2 = int((x_c_abs + width_abs / 2))
                y2 = int((y_c_abs + height_abs / 2))

                info.append((det_class, x1, y1, x2, y2))

        return info
    
class_names = ['Pit_Asbestos', 'Pit_Plastic', 'Manhole_Open']

def process_files(txt_directory, img_directory, pred=True):

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
           
                
        
                
                
txt_directory_pred = r'C:\Users\vishal.ponuganti\Downloads\Pit-Manhole_Open_predictions_with_txtfiles\detections_txt'
img_directory_pred = r'C:\Users\vishal.ponuganti\Downloads\Pit-Manhole_Open_predictions_with_txtfiles\detections_images'
pred_boxes = process_files(txt_directory_pred,img_directory_pred)


txt_directory_gt = r'C:\Users\vishal.ponuganti\Downloads\Pit-Manhole_Open Inference Images\Pit-Manhole_Open Inference Ground Truth'
img_directory_gt = r'C:\Users\vishal.ponuganti\Downloads\Pit-Manhole_Open Inference Images\Pit-Manhole_Open Inference Images'
gt_boxes = process_files(txt_directory_gt,img_directory_gt,pred=False)



class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
        self.TN_count = 0 

    def compute_iou(self, box1, box2):
        """
        Compute the IoU between two bounding boxes.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area_box1 + area_box2 - intersection

        return intersection / union if union > 0 else 0

    def update(self, pred_boxes_with_names, gt_boxes_with_names):
        """
        Update the confusion matrix with predicted and ground truth boxes.

        Parameters
        ----------
        pred_boxes_with_names : list
            List of tuples [(filename, [pred_boxes])], where pred_boxes are in the format 
            [x1, y1, x2, y2, confidence, class].
        gt_boxes_with_names : list
            List of tuples [(filename, [gt_boxes])], where gt_boxes are in the format 
            [class, x1, y1, x2, y2].
        """
        for pred_file, pred_boxes in pred_boxes_with_names:
            gt_boxes = next((g[1] for g in gt_boxes_with_names if g[0] == pred_file), [])

            gt_matched = set()
            pred_matched = set()

            if pred_boxes == [] and gt_boxes == []:
                self.TN_count += 1 
                with open(output_csv, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([pred_file, 'N/A', 'N/A', 'N/A','N/A','TN'])
                continue

            # Iterate through predictions
            for pred_idx, pred in enumerate(pred_boxes):
                pred_x1, pred_y1, pred_x2, pred_y2, conf, pred_class = pred
                if conf < self.CONF_THRESHOLD:
                    continue

                matched = False
                for gt_idx, gt in enumerate(gt_boxes):
                    gt_class, gt_x1, gt_y1, gt_x2, gt_y2 = gt
                    if gt_idx in gt_matched:
                        continue

                    iou = self.compute_iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
                    if iou >= self.IOU_THRESHOLD:
                        if pred_class == gt_class:
                            self.matrix[int(gt_class), int(pred_class)] += 1
                            gt_matched.add(gt_idx)
                            pred_matched.add(pred_idx)
                            matched = True
                        
                            with open(output_csv, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([pred_file, 
                                             f"{class_names[int(gt_class)]},({gt_x1},{gt_y1},{gt_x2},{gt_y2})",
                                             f"{class_names[int(pred_class)]},({pred_x1},{pred_y1},{pred_x2},{pred_y2})",
                                             conf,iou,'TP'])
                            break
                        else:
                            self.matrix[int(gt_class), int(pred_class)] += 1
                            gt_matched.add(gt_idx)
                            pred_matched.add(pred_idx)
                            matched = True

                            with open(output_csv, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([pred_file, 
                                             f"{class_names[int(gt_class)]},({gt_x1},{gt_y1},{gt_x2},{gt_y2})",
                                             f"{class_names[int(pred_class)]},{pred_x1},{pred_y1},{pred_x2},{pred_y2})",
                                             conf,iou,'FN'])


                if not matched:
                    # False positive
                    self.matrix[self.num_classes, int(pred_class)] += 1

                    with open(output_csv, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([pred_file, 'N/A',
                                         f"{class_names[int(pred_class)]},({pred_x1},{pred_y1},{pred_x2},{pred_y2})"
                                         ,conf,'N/A' ,'FP'])

            # Count unmatched ground truth boxes as false negatives
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx not in gt_matched:
                    gt_class, _, _, _, _ = gt
                    self.matrix[int(gt_class), self.num_classes] += 1

                    with open(output_csv, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([pred_file,
                                         f"{class_names[int(gt_class)]},({gt_x1},{gt_y1},{gt_x2},{gt_y2})",
                                          'N/A', 'N/A', 'N/A' ,'FN'])


    def calculate_metrics(self):
        """
        Calculate precision, recall, and accuracy for each class.
        """
        precisions = []
        recalls = []
        accuracies = []
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

        return precisions, recalls, accuracies
    
    def calculate_metrics_t(self):
        """
        Calculate precision, recall, and accuracy for each class.
        """
        precisions_total = []
        recalls_total = []
        accuracies_total = []
       
        TP_t = np.sum(np.diag(self.matrix))  # Sum of all True Positives
        FP_t = np.sum(self.matrix[self.num_classes, :self.num_classes])
        FN_t = np.sum(self.matrix) - (TP_t + FP_t)
        TN_t = self.TN_count

        precision_t = TP_t / (TP_t + FP_t) if TP_t + FP_t > 0 else 0
        recalls_t = TP_t / (TP_t + FN_t) if TP_t + FN_t > 0 else 0
        accuracy_t = (TP_t + TN_t) / (TP_t + TN_t + FP_t + FN_t) if TP_t + TN_t + FP_t + FN_t > 0 else 0


        return TP_t, FP_t, FN_t, TN_t 
    
   

    def plot(self, class_names):
    
        import matplotlib.pyplot as plt

        # Add an extra row and column for "FP" and "FN" labels
    

        x_labels = copy.deepcopy(class_names)
        y_labels = copy.deepcopy(class_names)
        x_labels.append('FN')
        y_labels.append('FP')

        # Create a heatmap for the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.matrix, annot=True, fmt='.0f', cmap='Blues', cbar=True,
                xticklabels=x_labels, yticklabels=y_labels, ax=ax)

        # Set axis labels
        ax.set_xlabel('Predicted Classes', fontsize=12)
        ax.set_ylabel('True Classes', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)

        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

class_names = ['Pit_Asbestos', 'Pit_Plastic', 'Manhole_Open']


# Update confusion matrix with predictions and ground truth

conf_matrix = ConfusionMatrix(num_classes=len(class_names))

#creating a timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#writing to an output csv file
output_csv = r"C:\Users\vishal.ponuganti\Downloads\Pit-Manhole_Open Inference Images"+"\Pit"+timestamp+".csv"
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Groundtruth', 'Prediction', 'Confidence','IoU', 'TP/FN/FP'])

#invoking the update function 
conf_matrix.update(pred_boxes, gt_boxes)

#printing confusion matrix
print("Confusion Matrix:\n", conf_matrix.matrix)

#calling function that calculate metrics
precisions, recalls, accuracies = conf_matrix.calculate_metrics()


TP_t, FP_t, FN_t, TN_t  = conf_matrix.calculate_metrics_t()

with open(output_csv, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([])  # Empty row for spacing
    writer.writerow(['True Positives:',TP_t])
    writer.writerow(['True Negatives:',TN_t])
    writer.writerow(['False Positives:',FP_t])
    writer.writerow(['False Negatives:',FN_t])

#writing csv file with class precision recall and accuracy
with open(output_csv, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([])  # Empty row for spacing
    writer.writerow(['Class', 'Precision', 'Recall', 'Accuracy'])

    for i, class_name in enumerate(class_names):
        writer.writerow([class_name, f"{precisions[i]:.2f}", f"{recalls[i]:.2f}", f"{accuracies[i]:.2f}"])

print("\nClass-wise Metrics:")
for i, class_name in enumerate(class_names):
    print(f"Class: {class_name}")
    print(f"  Precision: {precisions[i]:.2f}")
    print(f"  Recall: {recalls[i]:.2f}")
    print(f"  Accuracy: {accuracies[i]:.2f}")
    print("-------------------------------------")

# Print and plot the confusion matrix
conf_matrix.plot(class_names)

