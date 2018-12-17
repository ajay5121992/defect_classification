# defect_classification

This repository works on classifiying and detecting defects on a tool.
The repository uses keras-retinanet library for training and finding out the defect areas in image.
The tool use for bounding boxes/annotations is label-img
And repository uses pre train Resent_50 model which is trained on coco dataset for transfer learning.



The convert_xml_into_csv.py:-

Is use for converting the XML files into csv with format as [filname,xmin,ymin,xmax,ymax,class]
the converted csv is used for training and validation, for making csv's train.csv,validation.csv,classes.csv[class_name,class_id]


Train.py:-

Is the retinanet training file which is present in keras-retinanet/keras_retinanet/bin/train.py and used for object-detection training, the file takes command line argument with special argument as:

'annotations', help='Path to CSV file containing annotations for training.'

'classes', help='Path to a CSV file containing class label mapping.'

'--val-annotations', help='Path to CSV file containing annotations for validation (optional).'

'--weights',           help='Initialize the model with weights from a file.',
default = "/home/ajay/PycharmProjects/object_detection/resnet50_coco_best_v2.0.1.h5"

'--batch-size',       help='Size of the batches.', default=1, type=int

'--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str

'--steps',            help='Number of steps per epoch.', type=int, default=106

'--batch-size',       help='Size of the batches.', default=1, type=int

Image_agumentation.py:- 

Is used for data/image agumentation as noramlly the images supplied is not greater in number for deep learning so we uses DataAugmentationForObjectDetection library for image agumentation which conatins agumentation functions such as rotation, horizontal,vertical flip,sheared which takes the image with the bounding boxes array and returns the transformed image and new transformed bounding array's this way the data agumentation happens
