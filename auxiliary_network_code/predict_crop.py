import os
import glob
import torch
import cv2
import numpy as np
from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def save_detection_results(output_dir, image_paths, predictions, save_labels=False):
    no_detections_dir = os.path.join(output_dir, "No_Detections")
    os.makedirs(no_detections_dir, exist_ok = True)
    multiple_detections_dir = os.path.join(output_dir, "Multi_Detections")
    os.makedirs(multiple_detections_dir, exist_ok = True)
    single_detections_dir = os.path.join(output_dir, "Detections")
    os.makedirs(single_detections_dir, exist_ok = True)

    for index, (image_path, result) in enumerate(zip(image_paths, predictions)):
        image_filename = os.path.splitext(image_path.split("/")[-1])[0]
        class_dict = result.names

        if result.boxes.conf.size()[0] == 0:
            print(f"{image_filename} --> No Detections!\n")
            result.save(f"{no_detections_dir}/{image_filename}.JPG")
            continue
        
        array_of_confidences = result.boxes.conf.cpu().numpy()
        array_of_labels = result.boxes.cls.cpu().numpy()
        array_of_norm_coordinates = result.boxes.xywhn.cpu().numpy()
        max_conf_index = np.argmax(array_of_confidences)

        selected_label = int(array_of_labels[max_conf_index])
        selected_conf = array_of_confidences[max_conf_index]
        selected_norm_coordinates = array_of_norm_coordinates[max_conf_index]

        if len(array_of_labels) > 1:
            result.save(f"{multiple_detections_dir}/{image_filename}.JPG")
        
        else:
            save_image_path = os.path.join(single_detections_dir, "images")
            if selected_label == 7:
                stoat_path = os.path.join(save_image_path, "Stoat")
                os.makedirs(stoat_path, exist_ok = True)
                save_cropped_image(stoat_path, image_path, selected_norm_coordinates)
                # result.save(f"{stoat_path}/{image_filename}.JPG")
            else:
                others_path = os.path.join(save_image_path, "Others")
                os.makedirs(others_path, exist_ok = True)
                result.save(f"{others_path}/{image_filename}.JPG")

            if save_labels:
                save_label_path = os.path.join(single_detections_dir, "labels")
                os.makedirs(save_label_path, exist_ok = True)
                save_annotated_labels(save_label_path, image_filename, selected_label, selected_norm_coordinates)
            

def save_annotated_labels(output_dir, image_filename, label, norm_coordinates):
    if label == 7:
        label_dir = os.path.join(output_dir, "Stoat")
    else:
        label_dir = os.path.join(output_dir, "Others")
    os.makedirs(label_dir, exist_ok = True)

    text = str(label) + " " + " ".join(map(str, norm_coordinates))
    text_filename = image_filename + ".txt"
    text_path = os.path.join(label_dir, text_filename)
    with open(text_path, 'w') as file:
        file.write(text)


def filter_detection_results(output_dir, image_paths, predictions, conf_threshold):
    output_path = os.path.join(output_dir, "Detections", "High")
    os.makedirs(output_path, exist_ok = True)

    for index, (image_path, result) in enumerate(zip(image_paths, predictions)):
        image_filename = os.path.splitext(image_path.split("/")[-1])[0]
        class_dict = result.names
        array_of_confidences = result.boxes.conf.cpu().numpy()
        array_of_labels = result.boxes.cls.cpu().numpy()
        array_of_norm_coordinates = result.boxes.xywhn.cpu().numpy()

        if len(array_of_confidences) == 0:
            continue

        if array_of_confidences[0] >= conf_threshold:
            print(f"{image_filename} ({array_of_labels[0]}: {array_of_confidences[0]})\n")
            norm_bbox = array_of_norm_coordinates[0]
            save_cropped_image(output_dir, image_path, norm_bbox)
            # result.save(f"{output_path}/{image_filename}.JPG")
        

def save_cropped_image(output_dir, image_path, norm_bbox):
    # output_dir = "/raid/ywu840/NewZealandData/Stoat/DoC/All_Pens"
    # output_path = os.path.join(output_dir, "ReID")
    output_path = output_dir
    os.makedirs(output_path, exist_ok = True)

    image = cv2.imread(image_path)
    height, width, _ = image.shape
    x_center, y_center, bbox_width, bbox_height = norm_bbox    # normalized coordinates

    # Convert normalized coordinates to pixel coordinates.
    x_center = int(x_center * width)
    y_center = int(y_center * height)
    bbox_width = int(bbox_width * width)
    bbox_height = int(bbox_height * height)
        
    x_min = int(x_center - bbox_width / 2)
    y_min = int(y_center - bbox_height / 2)
    x_max = int(x_center + bbox_width / 2)
    y_max = int(y_center + bbox_height / 2)

    x_min, y_min, x_max, y_max = round(x_min), round(y_min), round(x_max), round(y_max)

    # Save cropped image.
    cropped_image = image[y_min:y_max, x_min:x_max, :]
    image_filename = image_path.split("/")[-1]
    cropped_image_path = os.path.join(output_path, image_filename)
    cv2.imwrite(cropped_image_path, cropped_image)




def main():
    DEVICE = torch.device("cuda")
    batch_size = 256
    # yolo = YOLO("Finetune_best.pt").to(DEVICE)
    yolo = YOLO("/data/yil708/Meta_Data/MetaData/auxiliary_network_code/Finetune_best.pt").to(DEVICE)

    # Construct the source path.
    # root = "/raid/ywu840/NewZealandData/Stoat/Stoat_Coal"
    root = "/data/yil708/Meta_Data/MetaData/auxiliary_network_code/"
    # source_dir = "Images"
    source_dir = "/data/yil708/Meta_Data/MetaData/auxiliary_network_code/auxiliary_network_pics/labelled_auxiliary_network_pics"
    source_path = os.path.join(root, source_dir)
    
    # Construct the destination path.
    # destination_dir = "Images_Results"
    destination_dir = "/data/yil708/Meta_Data/MetaData/auxiliary_network_code/auxiliary_network_pics/cropped_labelled_anpics"
    destination_path = os.path.join(root, destination_dir)
    os.makedirs(destination_path, exist_ok = True)

    
    image_paths_list = sorted(glob.glob(os.path.join(source_path, "*.JPG")) + glob.glob(os.path.join(source_path, "*.jpg")))
    num_of_images = len(image_paths_list)
    print(f"{num_of_images} Images\n")
    
    
    
    counter = 0
    while counter < num_of_images:
        process_images = image_paths_list[counter:min(counter + batch_size, num_of_images)]
        print(f"[{counter} - {min(counter + batch_size, num_of_images)}]")
        preds = yolo(process_images)
        print()
        save_detection_results(output_dir = destination_path, image_paths = process_images, predictions = preds, save_labels = False)
        counter += batch_size

    print(f"\nFinished {num_of_images} images!\n")
    
            
main()