## YOLO Algorithm for Object Detection

[![Issues](https://img.shields.io/github/issues/dabasajay/YOLO-Object-Detection.svg?color=%231155cc)](https://github.com/dabasajay/YOLO-Object-Detection/issues)
[![Forks](https://img.shields.io/github/forks/dabasajay/YOLO-Object-Detection.svg?color=%231155cc)](https://github.com/dabasajay/YOLO-Object-Detection/network)
[![Stars](https://img.shields.io/github/stars/dabasajay/YOLO-Object-Detection.svg?color=%231155cc)](https://github.com/dabasajay/YOLO-Object-Detection/stargazers)
[![Ajay Dabas](https://img.shields.io/badge/Ajay-Dabas-ff0000.svg)](https://dabasajay.github.io/)

In this project, I use YOLO Algorithm trained on COCO Dataset for object detection. I use pretrained Yolov2 model which can downloaded from the official YOLO <a href='https://pjreddie.com/darknet/yolo/'>website</a>.

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
	<li>A good CPU and a GPU with atleast 8GB memory</li>
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

## How to Use

Just follow 5 simple steps :

1. Download Darknet model cfg and weights from the official YOLO <a href='https://pjreddie.com/darknet/yolo/'>website</a> and put them in `model_data/` folder.<br>
**Direct commands**<br>
`wget https://pjreddie.com/media/files/yolov2.weights`<br>
`mv yolov2.weights "model_data/yolov2.weights`<br>
`wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg`<br>
`mv yolov2.cfg "model_data/yolov2.cfg"`
2. Review `config.py` for paths and other configurations (explained below)
3. Run `yad2k.py` to convert YOLOv2 model from darknet to keras model which will be saved in `model_data/` folder.
4. Put all your images you want to test in `images/` directory.
5. Run `YOLO.py` and MAGIC! Output images will be saved in `output/` directory

Script `yad2k.py` for conversion of darknet to keras model is copied from <strong>Allan Zelener - </strong><a href='https://github.com/allanzelener/YAD2K'>YAD2K: Yet Another Darknet 2 Keras</a> github repo and modified a little bit.

## Configurations (config.py)

1. `keras_model_path` :- Folder path containing keras converted yolov2 model in .hdf5 format
2. `model_weights` :- Folder path containing yolov2 weights file from darknet (Downloaded file)
3. `model_cfg` :- Folder path containing yolov2 configuration file from darknet (Downloaded file)
4. `anchors_path` :- Folder path containing yolo_anchors.txt
5. `classes_path` :- Folder path containing coco_classes.txt
6. `test_path` :- Folder path containing images for testing model
7. `output_path` :- Folder path containing output of images from test_path
8. `score_threshold` :- Score(Confidence of predicted class) threshold. Lower value leads to more classes predictions but less confident about predictions, higher leads to less classes predictions but more confident about predictions.
9. `iou_threshold` :- Intersection over union threshold.
10. `random_seed` :- Random seed for reproducibility of results
11. `font_path` :- Folder path containing font to write on bounding boxes in image

## References

The ideas presented in this repo came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from **Allan Zelener**'s github repository. The pretrained weights used in this project came from the official YOLO website.

<ul type='square'>
  <li><strong>Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - </strong><a href='https://arxiv.org/abs/1506.02640'>You Only Look Once: Unified, Real-Time Object Detection (2015)</a></li>
  <li><strong>Joseph Redmon, Ali Farhadi - </strong><a href='https://arxiv.org/abs/1612.08242'>YOLO9000: Better, Faster, Stronger (2016)</a></li>
  <li><strong>Allan Zelener - </strong><a href='https://github.com/allanzelener/YAD2K'>YAD2K: Yet Another Darknet 2 Keras</a></li>
  <li><a href='https://pjreddie.com/darknet/yolo/'><strong>The official YOLO website</strong></a></li>
</ul>
