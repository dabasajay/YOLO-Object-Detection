""" Run a YOLO_v2 style detection model on test images. """
# Libraries
import colorsys
import imghdr
import os
import random
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import yolo_eval, yolo_head
# Configuration
from config import config

def _main():
    # Paths
    model_path = os.path.expanduser(config['keras_model_path'])
    assert model_path.endswith('.hdf5'), 'Keras model must be a .hdf5 file.'
    anchors_path = os.path.expanduser(config['anchors_path'])
    classes_path = os.path.expanduser(config['classes_path'])
    test_path = os.path.expanduser(config['test_path'])
    output_path = os.path.expanduser(config['output_path'])
    # Check if output path exists
    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)
    # Get tensorflow session
    sess = K.get_session()
    # Read yolov2 classes
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    # Read yolov2 anchors
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    # Load yolov2 model
    yolo_model = load_model(model_path)
    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), 'Mismatch between model and given anchor and class sizes.'
    print('{} model, anchors, and classes loaded.'.format(model_path))
    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list( map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
    # Fixed seed for consistent colors across runs.
    random.seed(config['random_seed'])
    # Shuffle colors to decorrelate adjacent classes.
    random.shuffle(colors)
    # Generate output tensor targets for filtered bounding boxes.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=config['score_threshold'],
        iou_threshold=config['iou_threshold']
        )
    # For each image file given in `test_path`
    for image_file in os.listdir(test_path):
        try:
            # Read file type
            image_type = imghdr.what(os.path.join(test_path, image_file))
            if not image_type:
                # If file is not an image type
                continue
        except IsADirectoryError:
            # Image not found
            continue
        # If file is correct image type, read it into array
        image = Image.open(os.path.join(test_path, image_file))
        # If model is fully convolutional
        if is_fixed_size:
            resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        # Normalize image
        image_data /= 255.
        # Add batch dimension.
        image_data = np.expand_dims(image_data, 0)
        # Run tensorflow session to execute the graph and get results
        out_boxes, out_scores, out_classes = sess.run(
            fetches=[boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))
        # Load font to write on bounding boxes in image
        font = ImageFont.truetype(
            font=config['font_path'],
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        # For each class detected in image
        for i, c in reversed(list(enumerate(out_classes))):
            # Predicted class
            predicted_class = class_names[c]
            # Bounding box coordinates
            box = out_boxes[i]
            # Confidence score
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            # Exract bounding box coordinates
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        # Save image to `output_path`
        image.save(os.path.join(output_path, 'output-'+image_file), quality=90)
    # Close tensorflow session
    sess.close()

if __name__ == '__main__':
    _main()