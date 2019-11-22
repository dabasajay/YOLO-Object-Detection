## YOLO Algorithm for Object Detection

[![Issues](https://img.shields.io/github/issues/dabasajay/YOLO-Object-Detection.svg?color=%231155cc)](https://github.com/dabasajay/YOLO-Object-Detection/issues)
[![Forks](https://img.shields.io/github/forks/dabasajay/YOLO-Object-Detection.svg?color=%231155cc)](https://github.com/dabasajay/YOLO-Object-Detection/network)
[![Stars](https://img.shields.io/github/stars/dabasajay/YOLO-Object-Detection.svg?color=%231155cc)](https://github.com/dabasajay/YOLO-Object-Detection/stargazers)
[![Ajay Dabas](https://img.shields.io/badge/Ajay-Dabas-825ee4.svg)](https://dabasajay.github.io/)

In this project, I used YOLO algorithm trained on COCO dataset for object detection task. I used pretrained Yolov2 model which can downloaded from the official YOLO <a href='https://pjreddie.com/darknet/yolo/'>website</a>.

<p align="center">
  <strong>Examples</strong>
</p>

<p align="center">
  <img src="output/Steffen Muldbjerg.jpg?raw=true" height="400px" width="50%" title="Photo by Steffen Muldbjerg on Unsplash" alt="Photo by Steffen Muldbjerg on Unsplash">
  <img src="output/Alexander McFeron.jpg?raw=true" height="400px" width="50%" title="Photo by Alexander McFeron on Unsplash" alt="Photo by Alexander McFeron on Unsplash">
</p>

## Requirements

Recommended System Requirements to run model.

<ul type="square">
	<li>A good CPU and a GPU with atleast 4GB memory</li>
	<li>Atleast 8GB of RAM</li>
	<li>Active internet connection to download YOLOv2 weights and cfg file.</li>
</ul>

Required Libraries for Python along with their version numbers used while making & testing of this project

<ul type="square">
	<li>Python - 3.6.7</li>
	<li>Numpy - 1.16.4</li>
	<li>Tensorflow - 1.13.1</li>
	<li>Keras - 2.2.4</li>
	<li>PIL - 4.3.0</li>
</ul>

## Some YOLOv2 Model information

1. Total params: 50,983,561
2. Trainable params: 50,962,889
3. Non-trainable params: 20,672

## How to Use

Just follow 6 simple steps :

1. Clone repository to preserve directory structure<br>
`git clone https://github.com/dabasajay/YOLO-Object-Detection.git`
2. Download Darknet model cfg and weights from the official YOLO <a href='https://pjreddie.com/darknet/yolo/'>website</a> and put them in `model_data/` folder.<br>
**Direct terminal commands**<br>
`wget https://pjreddie.com/media/files/yolov2.weights`<br>
`mv yolov2.weights "model_data/yolov2.weights"`<br>
`wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov2.cfg`<br>
`mv yolov2.cfg "model_data/yolov2.cfg"`
3. Review `config.py` for paths and other configurations (explained below)
4. Run `yad2k.py` to convert YOLOv2 model from darknet to keras model which will be saved in `model_data/` folder.
5. Put all your images you want to test in `images/` directory.<br>
**Note:** All images are resized to 608x608 to feed into YOLOv2 model
6. Run `YOLO.py` and *MAGIC!* Output images will be saved in `output/` directory

**If you face any problem converting keras model or anything:** email me at *se.dabasajay@gmail.com*

**Acknowledgement:** Script `yad2k.py` for conversion of darknet to keras model is taken from <strong>Allan Zelener - </strong><a href='https://github.com/allanzelener/YAD2K'>YAD2K: Yet Another Darknet 2 Keras</a> github repo and modified a little bit.

## Configurations (config.py)

1. `keras_model_path` :- File path of keras converted yolov2 model in .hdf5 format
2. `model_weights` :- File path of yolov2 weights file from darknet (Downloaded file)
3. `model_cfg` :- File path of yolov2 configuration file from darknet (Downloaded file)
4. `anchors_path` :- File path of yolo_anchors.txt
5. `classes_path` :- File path of coco_classes.txt
6. `test_path` :- Folder path containing images for testing model
7. `output_path` :- Folder path containing output of images from test_path
8. `score_threshold` :- Score(Confidence of predicted class) threshold. Lower value leads to more class predictions but less confident about predictions, higher leads to less class predictions but more confident about predictions.
9. `iou_threshold` :- Intersection over union threshold.
10. `random_seed` :- Random seed for reproducibility of results
11. `font_path` :- File path of font to write on bounding boxes in image

## References

The ideas presented in this repo came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from **Allan Zelener**'s github repository. The pretrained weights used in this project came from the official YOLO website.

<ul type='square'>
  <li><strong>Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - </strong><a href='https://arxiv.org/abs/1506.02640'>You Only Look Once: Unified, Real-Time Object Detection (2015)</a></li>
  <li><strong>Joseph Redmon, Ali Farhadi - </strong><a href='https://arxiv.org/abs/1612.08242'>YOLO9000: Better, Faster, Stronger (2016)</a></li>
  <li><strong>Allan Zelener - </strong><a href='https://github.com/allanzelener/YAD2K'>YAD2K: Yet Another Darknet 2 Keras</a></li>
  <li><a href='https://pjreddie.com/darknet/yolo/'><strong>The official YOLO website</strong></a></li>
</ul>
