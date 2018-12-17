from DataAugmentationForObjectDetection.data_aug.data_aug import *
from DataAugmentationForObjectDetection.data_aug.bbox_util import *
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
# matplotlib inline
img = cv2.imread("/home/ajay/YE358311_Fender_apron_2/YE358311_Fender_apron/YE358311_defects/"
                 "YE358311_Crack_and_Wrinkle_defect/IMG20180905150333.jpg")[:,:,::-1]   #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
base_path_images ="/home/ajay/YE358311_Fender_apron_2/YE358311_Fender_apron/new_agumented_images/"

defected_tool_csv = pd.read_csv("/home/ajay/PycharmProjects/object_detection/defected_tool_labels.csv")

new_list_of_csv = []
for index, row in defected_tool_csv.iterrows():
    try:
        RandomHorizontalFlip_0 = []
        RandomHorizontalFlip_1 = []
        RandomScale_ = []
        RandomTranslate_ = []
        RandomRotate_ = []
        RandomShear_ = []
        RandomHSV_ = []
        seq_ = []

        cords = np.array([[row['xmin'], row['ymin'], row['xmax'], row['ymax']]], np.float64)
        class_ = row['class']
        image_name = row['filename']
        img = cv2.imread("/home/ajay/YE358311_Fender_apron_2/YE358311_Fender_apron/"
                         "YE358311_defects/YE358311_Crack_and_Wrinkle_defect/" + image_name)[:, :,
              ::-1]  # opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb

        img_, bboxes_ = RandomHorizontalFlip(0)(img.copy(), cords)
        width = img_.shape[1]
        height = img_.shape[0]
        RandomHorizontalFlip_0.extend(
            [os.path.splitext(image_name)[0] + "RandomHorizontalFlip_0.jpg", width, height, class_,
             bboxes_[0][0], bboxes_[0][1], bboxes_[0][2], bboxes_[0][3]])
        # im = Image.fromarray(img_)
        # im.save(base_path_images + os.path.splitext(image_name)[0] + "RandomHorizontalFlip_0.jpg")

        img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), cords)
        RandomHorizontalFlip_1.extend(
            [os.path.splitext(image_name)[0] + "RandomHorizontalFlip_1.jpg", width, height, class_,
             bboxes_[0][0], bboxes_[0][1], bboxes_[0][2], bboxes_[0][3]])
        # im = Image.fromarray(img_)
        # im.save(base_path_images + os.path.splitext(image_name)[0] + "RandomHorizontalFlip_1.jpg")

        img_, bboxes_ = RandomScale(0.3, diff=True)(img, cords)
        RandomScale_.extend([os.path.splitext(image_name)[0] + "RandomScale.jpg", width, height, class_,
                             bboxes_[0][0], bboxes_[0][1], bboxes_[0][2], bboxes_[0][3]])
        # im = Image.fromarray(img_)
        # im.save(base_path_images + os.path.splitext(image_name)[0] + "RandomScale.jpg")

        img_, bboxes_ = RandomTranslate(0.3, diff=True)(img, cords)
        RandomTranslate_.extend([os.path.splitext(image_name)[0] + "RandomTranslate.jpg", width, height, class_,
                                 bboxes_[0][0], bboxes_[0][1], bboxes_[0][2], bboxes_[0][3]])
        # im = Image.fromarray(img_)
        # im.save(base_path_images + os.path.splitext(image_name)[0] + "RandomTranslate.jpg")

        img_, bboxes_ = RandomRotate(20)(img.copy(), cords)
        RandomRotate_.extend([os.path.splitext(image_name)[0] + "RandomRotate.jpg", width, height, class_,
                              bboxes_[0][0], bboxes_[0][1], bboxes_[0][2], bboxes_[0][3]])
        # im = Image.fromarray(img_)
        # im.save(base_path_images + os.path.splitext(image_name)[0] + "RandomRotate.jpg")

        img_, bboxes_ = RandomShear(0.2)(img.copy(), cords)
        RandomShear_.extend([os.path.splitext(image_name)[0] + "RandomShear.jpg", width, height, class_,
                             bboxes_[0][0], bboxes_[0][1], bboxes_[0][2], bboxes_[0][3]])
        # im = Image.fromarray(img_)
        # im.save(base_path_images + os.path.splitext(image_name)[0] + "RandomShear.jpg")

        img_, bboxes_ = RandomHSV(100, 100, 100)(img.copy(), cords)
        RandomHSV_.extend([os.path.splitext(image_name)[0] + "RandomHSV.jpg", width, height, class_,
                           bboxes_[0][0], bboxes_[0][1], bboxes_[0][2], bboxes_[0][3]])
        # im = Image.fromarray(img_)
        # im.save(base_path_images + os.path.splitext(image_name)[0] + "RandomHSV.jpg")

        seq = Sequence([RandomHSV(40, 40, 30), RandomHorizontalFlip(),
                        RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()])
        img_, bboxes_ = seq(img.copy(), cords)
        seq_.extend([os.path.splitext(image_name)[0] + "seq.jpg", width, height, class_,
                     bboxes_[0][0], bboxes_[0][1], bboxes_[0][2], bboxes_[0][3]])
        # im = Image.fromarray(img_)
        # im.save(base_path_images + os.path.splitext(image_name)[0] + "seq.jpg")
        new_list_of_csv.append(RandomHorizontalFlip_0)
        new_list_of_csv.append(RandomHorizontalFlip_1)
        new_list_of_csv.append(RandomScale_)
        new_list_of_csv.append(RandomTranslate_)
        new_list_of_csv.append(RandomRotate_)
        new_list_of_csv.append(RandomShear_)
        new_list_of_csv.append(RandomHSV_)
        new_list_of_csv.append(seq_)
    except Exception as e:
        print(e)

new_df=pd.DataFrame(new_list_of_csv,columns=["filename","width","height","class",
                                             "xmin","ymin","xmax","ymax"])
new_df.to_csv("new_defected_tool.csv")