# defect_classification

This repository works on classifiying and detecting defects on a tool. it uses keras-retinanet library for training and finding out the defect areas in image and classifiying it.




The process as of this project starts as of by taking the images and forming bounding boxes around the objects/points which needs to be detect.


The tool use for bounding boxes/annotations is label-img(https://github.com/tzutalin/labelImg). Its graphical image annotation tool annotations are saved as XML files in PASCAL VOC format, the format used by ImageNet. 
Xml files for each image contains cordinates of specified object of certain location with filename,class name associated with object. Now we need to convert the xmls into csv to work on, the code for this as follows

The convert_xml_into_csv.py:-
Is use for converting the XML files into csv with format as [filname,xmin,ymin,xmax,ymax,class]
the converted csv is used for training and validation dataset, for making csv's train.csv,validation.csv,classes.csv[class_name,class_id]




As the images are less in number that would not be good for deep learning so we would have to use some data agumentation techquies 

Image_agumentation.py:- 

Is used for data/image agumentation as noramlly the images supplied is not greater in number for deep learning so we uses DataAugmentationForObjectDetection(https://github.com/Paperspace/DataAugmentationForObjectDetection) library for image agumentation which conatins agumentation functions such as rotation, horizontal,vertical flip,sheared which takes the image with the bounding boxes array and returns the transformed image and new transformed bounding array's this way the data agumentation happens




The training of object detection is done by using the retinanet's keras-retinanet training file name train.py the file and we use pre train Resent_50 model which is trained on coco dataset for transfer learning.

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



predict.pyis used for predicting the result of model trained by running it trained model,for whuch we have to convert the trained model into inference model

![Screenshot](example_test_defect.png)
