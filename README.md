#  3DSegStyle: 3D Style Transfer on Any Object in the Scene


This repository contains a final project for Peking University's 2024 spring semester course "Frontiers of Geometric Computing", introduced by [Prof. Peng-Shuai Wang](https://wang-ps.github.io/). 

The main propose of this project is to achieve instant 3D style transfer on any object in the scene. We first train the scene with 3D gaussian splatting modeling, then perform segmentation on the trained gaussians, and finally use the segmentation results to achieve style transfer at specific objects. 

To achieve this, we utilize open source projects [StyleGaussian](https://arxiv.org/abs/2403.07807) and [SAGA (Segment Any 3D GAussians)](https://arxiv.org/abs/2312.00860). Thank them a lot!


[![Demo](https://res.cloudinary.com/marcomontalbano/image/upload/v1719566181/video_to_markdown/images/youtube--6S12YDWTCq8-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=6S12YDWTCq8 "Demo")

---

## 1 Installation

You can clone our environment directly by
```bash
pip install -r requirements.txt
```

Or you can also install the package in `requirements.txt` manually.

## 2 Quick Start

[Datasets and Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1xHGXniVL3nh6G7pKDkZR1SJlfvo4YB1J?usp=sharing)

Please download the pre-processed datasets and put them in the `datasets` folder. We also provide the pre-trained checkpoints, which should be put in the `output` folder (see `output/readme.md` for details). If you change the location of the datasets or the location of the checkpoints, please modify the `model_path` or `source_path` accordingly in the `cfg_args` in the checkpoint folder.

Please create two folders `SegAnyGAussians/data` and `SegAnyGAussians/pretrained_models`. Follow the guidance in `SegAnyGAussians/README.md` to initialize SAGA model. Put your data under `SegAnyGAussians/data`, and set your SAGA model path as `SegAnyGAussians/pretrained_models`. 

### 2.1 Interactive Remote Viewer

https://github.com/Kunhao-Liu/StyleGaussian/assets/63272695/d6dfda95-b272-42ff-b855-e16801f594a9

Our interactive remote viewer is based on [Viser](https://github.com/nerfstudio-project/viser). To start the viewer, run the following command:
```bash
python seg_style.py -m [model_path] --style_folder [style_folder] --viewer_port [viewer_port]

# for example:
python seg_style.py -m output/garden/artistic/default --style_folder images --viewer_port 5748
```
where `model_path` is the path to the pre-trained model, named as `output/[scene_name]/artistic/[exp_name]`, `style_folder` is the folder containing the style images, and `viewer_port` is the port number for the viewer. `--style_folder` and `--viewer_port` are optional arguments.

#### Usage:
 1. **Rendering Settings.** Use the mouse to rotate the camera, and use the mouse wheel to zoom in and out. Use `WASD` to move the camera, `QE` to move up and down. You can choose the viewpoint from the training views. Click `Reset View` to reset the up direction.  You can also change the rendering resolution and image quality which affect the rendering speed as the refrasing speed is limited by the internet speed. 
 2. **Style Transfer.** The rendering will start once you specify the style image path in `Style Image` text box and the style will be transferred instantly. The `Random Style` button will randomly select a style image from the style folder. Click the `Display Style Image` button to display the style image at the lower right corner.
 3. **Partially Transfer.** Only for `garden` scene. Click each object button (e.g. `Table`, `Flower`, `Floor`, ...) to display the style transfer result on these specific objects while other parts of scene remain unchanged.
 4. **Style Interpolation.** You can interpolate between two style images by specifying the image paths of the two style images in `Style 1` and `Style 2`. The interpolation ratio can be adjusted by the slider. The  `Style 1` image is displayed at the lower left corner and the `Style 2` image is displayed at the lower right corner.


### 2.2 Inference Rendering

#### 1. Style Transfer
You can perform style transfer on a scene with a single style image by:
```bash
python render.py -m [model_path] --style [style_image_path] 

# for example:
python render.py -m output/garden/artistic/default --style images/0.jpg
```
where `model_path` is the path to the pre-trained model, named as `output/[scene_name]/artistic/[exp_name]`, and `style_image_path` is the path to the style image. The rendered stylized multi-view images will be saved in the `output/[scene_name]/artistic/[exp_name]/train` folder.

#### 2. Style Interpolation
You can perform style interpolation on a scene with **four** style images by:
```bash
python render.py -m [model_path] --style [style_image_path1] [style_image_path2] [style_image_path3] [style_image_path4]

# for example:
python render.py -m output/garden/artistic/default --style images/0.jpg images/1.jpg images/2.jpg images/3.jpg
```
where `model_path` is the path to the pre-trained model, named as `output/[scene_name]/artistic/[exp_name]`, and `style_image_path1`, `style_image_path2`, `style_image_path3`, `style_image_path4` are the paths to the four style images. The rendered interpolated stylized visualizations will be saved in the `output/[scene_name]/artistic/[exp_name]/style_interpolation` folder.


## 3 Training
We use the [WikiArt Dataset](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan) as the style images dataset.

### 3.1 Train from Scratch
You can train the model from scratch by:
```bash
python train.py --data [dataset_path] --wikiartdir [wikiart_path] --exp_name [exp_name]

# for example:
python train.py --data datasets/garden --wikiartdir datasets/wikiart --exp_name default
```
where `dataset_path` is the path to the training dataset, `wikiart_path` is the path to the WikiArt dataset, and `exp_name` is the name of the experiment. The training process will be saved in the `output/[scene_name]/artistic/[exp_name]` folder. `--exp_name` is an optional argument.

### 3.2 Train Step by Step
Alternatively, you can train the model step by step. StyleGaussian is trained in three steps: *reconstruction training*, *feature embedding training*, and *style transfer training*. These three steps can be trained separately by:

#### 1. Reconstruction Training
```bash
python train_reconstruction.py -s [dataset_path]

# for example:
python train_reconstruction.py -s datasets/garden
```
The trained reconstruction model will be saved in the `output/[scene_name]/reconstruction` folder.

#### 2. Feature Embedding Training
```bash
python train_feature.py -s [dataset_path] --ply_path [ply_path]

# for example:
python train_feature.py -s datasets/garden --ply_path output/garden/reconstruction/default/point_cloud/iteration_30000/point_cloud.ply
```
where `dataset_path` is the path to the training dataset, and `ply_path` is the path to the 3D Gaussians reconstructed from the reconstruction training stage, name as `output/[scene_name]/reconstruction/[exp_name]/point_cloud/iteration_30000/point_cloud.ply`. The trained feature embedding model will be saved in the `output/[scene_name]/feature` folder.


#### 3. Style Transfer Training
```bash
python train_artistic.py -s [dataset_path] --wikiartdir [wikiart_path] --ckpt_path [feature_ckpt_path] --style_weight [style_weight] 

# for example:
python train_artistic.py -s datasets/garden --wikiartdir datasets/wikiart --ckpt_path output/garden/feature/default/chkpnt/feature.pth --style_weight 10
```
where `dataset_path` is the path to the training dataset, `wikiart_path` is the path to the WikiArt dataset, `feature_ckpt_path` is the path to the checkpoint of the feature embedding model, named as `output/[scene_name]/feature/[exp_name]/chkpnt/feature.pth`, and `style_weight` is the weight for the style loss. The trained style transfer model will be saved in the `output/[scene_name]/artistic/[exp_name]` folder. `--style_weight` is an optional argument.

### 3.3 Train with Initialization
You can save the training time of the style transfer training stage by initializing the model with the trained model of another scene by:
```bash
python train_artistic.py -s [dataset_path] --wikiartdir [wikiart_path] --ckpt_path [feature_ckpt_path] --style_weight [style_weight] --decoder_path [init_ckpt_path]

# for example:
python train_artistic.py -s datasets/train --wikiartdir datasets/wikiart --ckpt_path output/train/feature/default/chkpnt/feature.pth --style_weight 10 --decoder_path output/truck/artistic/default/chkpnt/gaussians.pth
```
where `dataset_path` is the path to the training dataset, `wikiart_path` is the path to the WikiArt dataset, `feature_ckpt_path` is the path to the checkpoint of the feature embedding model, named as `output/[scene_name]/feature/[exp_name]/chkpnt/feature.pth`, `style_weight` is the weight for the style loss, and `init_ckpt_path` is the path to the checkpoint of the trained model of another scene, named as `output/[another_scene]/artistic/[exp_name]/chkpnt/gaussians.pth`. The trained style transfer model will be saved in the `output/[scene_name]/artistic/[exp_name]` folder.

## 4 Acknowledgements

Our work is based on [StyleGaussian](https://arxiv.org/abs/2403.07807) and [SAGA (Segment Any 3D GAussians)](https://arxiv.org/abs/2312.00860). We thank the authors for their great work and open-sourcing the code.