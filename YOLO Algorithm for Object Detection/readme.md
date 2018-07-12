<h1>YOLO Algorithm for Object Detection</h1>
In this project, I use YOLO Algorithm trained on COCO Dataset for object detection. I use pretrained Yolov2 model which can downloaded from the official YOLO <a href='https://pjreddie.com/darknet/yolo/'>site</a>.
<h4>All the references and licenses are included in end of this page</h4>

<h1>How to Use</h1>
>Just follow 5 simple steps:
<ul type='square'>
  `code`
  <li>Download Darknet model cfg and weights from the official YOLO website and convert the Darknet YOLO_v2 model to a Keras model. This is by far the lengthiest step. :weary:</li>
  <li>Save the model in <strong>model_data/</strong> directory. :sparkles:</li>
  <li>Put all your images you want to test in <strong>images/</strong> directory. :camera:</li>
  <li>Run <strong>YOLO.py</strong> and MAGIC! :dizzy:</li>
  <li>Output images will be saved in <strong>output/</strong> directory. :metal:</li>
</ul>
>For more information, visit <a href='https://github.com/allanzelener/YAD2K'>here</a>.

<h1>References :page_facing_up:</h1>
The ideas presented in this repo came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from <strong>Allan Zelener</strong>'s github repository. The pretrained weights used in this exercise came from the official YOLO website.
<ul type='square'>
  <li><strong>Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - </strong><a href='https://arxiv.org/abs/1506.02640'>You Only Look Once: Unified, Real-Time Object Detection (2015)</a></li>
  <li><strong>Joseph Redmon, Ali Farhadi - </strong><a href='https://arxiv.org/abs/1612.08242'>YOLO9000: Better, Faster, Stronger (2016)</a></li>
  <li><strong>Allan Zelener - </strong><a href='https://github.com/allanzelener/YAD2K'>YAD2K: Yet Another Darknet 2 Keras</a></li>
  <li><a href='https://pjreddie.com/darknet/yolo/'><strong>The official YOLO website</strong></a></li>
</ul>
