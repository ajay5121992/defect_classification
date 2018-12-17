import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from keras.preprocessing.image import ImageDataGenerator,img_to_array
import cv2

images_path="/home/ajay/YE358311_Fender_apron_2/YE358311_Fender_apron/YE358311_defects/YE358311_Crack_and_Wrinkle_defect/"
hor_flipped_image = ImageDataGenerator(horizontal_flip=True)
ver_flipped_image = ImageDataGenerator(vertical_flip=True)

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            image_name = root.find('filename').text
            image = cv2.imread(images_path+image_name)
            x = img_to_array(image)
            x = x.reshape((1,) + x.shape)
            hor_flipped_image.flow(x, batch_size=1, save_to_dir=images_path, save_prefix="hor_flipped",save_format='jpeg')
            ver_flipped_image.flow(x, batch_size=1, save_to_dir=images_path, save_prefix="ver_flipped", save_format='jpeg')
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), '/home/ajay/YE358311_Fender_apron_2/YE358311_Fender_apron/object_annotated_images')
    print (image_path)
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('defected_tool_labels.csv', index=None)
    print('Successfully converted xml to csv.')


main()
