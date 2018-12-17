import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
model_path = os.path.join('new_model.h5')
model = models.load_model(model_path, backbone_name='resnet50')

labels_to_names = {0: 'crack', 1: 'wrinkle'}

c = 0
im_path = '/home/ajay/PycharmProjects/object_detection/keras-retinanet/keras_retinanet/bin/images'
save_path = 'marked_results'

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,font_scale=2, thickness=2):
    x, y = coordinates[:2]

    cv2.putText(image_array, text, (x + x_offset, y + y_offset),cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

# color = np.asarray(255, 255, 0).tolist()


for file in os.listdir(im_path):
    print(file)

    file_path = os.path.join(im_path, file)

    image = read_image_bgr(file_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score >= 0.50 and label in range(0, 1):
            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            # draw_caption(draw, b, caption)
            print(caption, b)
            # draw_text(box, draw, labels_to_names[label], [255,255,0], 0, -45, 1, 1)
            cv2.putText(draw, caption, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 5, cv2.LINE_8)
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(draw)
            plt.show()
            # plt.savefig("example_test_defect.png")
            cv2.imwrite('example_test_defect.png', draw)
    # draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    # l, b = draw.shape[:2]
    # ratio = 800 / l
    # cv2.imshow('0', cv2.resize(draw, None, fx=ratio, fy=ratio))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()