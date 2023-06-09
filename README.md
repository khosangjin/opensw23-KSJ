
# opensw23-KSJ

# Team Introduction
* Name : Khosangjin 
* ID :  202011248
* Role : overall in project

# Topic Introduction

* ## 딥 드림 (Deep-Dream, Neural-dream)
	* 딥 드림(Deep-Dream)은 합성곱-신경망(Convolutional neural network, CNN)이 학습한 패턴을 시각화(Visualize)한 알고리즘입니다.
	
	* 입력된 이미지를 의도적으로 패턴을 과잉 해석함으로써, hallucinate하고 몽환적인 느낌의 사진을 출력합니다.

# Results
  
* 기존 repo의 example 사진 포함 총 24개의 사진을 실험했습니다.
* 이중 대표적인 사진을 소개합니다
  
	* 원본
![원본](https://i.esdrop.com/d/f/XDglyqtPeL/4kJdlMe3t1.jpg "원본")

	* 결과 (GIF으로 출력함으로 결과물이 얻어지는 과정을 보여준다)
![gif](https://i.esdrop.com/d/f/XDglyqtPeL/v3fOmRy4sG.gif "gif")

* learning_rate(학습률) 값을 다르게 입력함으로써 사진이 출력되는 결과입니다.
	
	* 학습률 0.5 적용
![학습률 0.5 적용](https://i.esdrop.com/d/f/XDglyqtPeL/Sz5ZQtKTDl.png "학습률 0.5 적용")
	
	* 학습률 1.5 적용( default)
![학습률 1.5 적용(default)](https://i.esdrop.com/d/f/XDglyqtPeL/qheN0WnrGC.png "학습률 1.5 적용")

	* 학습률 3.0 적용
![학습률 3.0 적용](https://i.esdrop.com/d/f/XDglyqtPeL/B6WHYGEB7V.png "학습률 3.0 적용")

* 다른 다양한 사진
<p align="center">	
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/6pvp7CzRP2.png" align="center" width="45%">
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/B9LUi8909F.png" align="center" width="45%">
	<figcaption align="center">제주바다</figcaption>
</p>

<p align="center">	
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/hOEDD52ozN.png" align="center" width="45%">
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/4nhlHislvY.png" align="center" width="45%">
	<figcaption align="center">동해바다</figcaption>
</p>

<p align="center">	
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/HRFS5rSwkB.png" align="center" width="45%">
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/NHRxtUNKX8.png" align="center" width="45%">
	<figcaption align="center">한라산</figcaption>
</p>

<p align="center">	
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/WJyCV1ibbU.png" align="center" width="45%">
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/ts7kNnYYJk.png" align="center" width="45%">
	<figcaption align="center">임진강</figcaption>
</p>

<p align="center">	
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/yvA5GYKkvt.png" align="center" width="45%">
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/aUYohcnzo0.png" align="center" width="45%">
	<figcaption align="center">HCTR</figcaption>
</p>

# Analysis/Visualization

* 여러 이미지들을 deep-dream한 결과물들 간의 유사성

### 1. 원본 이미지에서 하늘이나 바닥같이 색 변화나 패턴이 없는 구역에서 특정 패턴이 등장하는 것이 확인됨.

* 특성을 더 면밀히 관찰하기 위해 단색의 이미지로 실험을 진행함.

<p align="center">	
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/GtPCwHwNEo.png" align="center" width="90%">
	<figcaption align="center">원본 이미지</figcaption>
</p>


<p align="center">
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/6HCTcX7ijP.png" align="center" width="90%">
	<figcaption align="center">1회 반복</figcaption>
</p>
	
	
<p align="center">
	<img src="https://i.esdrop.com/d/f/XDglyqtPeL/NRtkouq7H4.png" align="center" width="90%">
	<figcaption align="center">10회 반복</figcaption>
</p>

* 단색 이미지임에도 불구하고, 출력된 이미지에서 구불구불한 문양과 일부 부분에서는 동물과 탑 모양의 패턴을 찾을 수 있음.
* 이러한 패턴이 발생한 이유는 inception 모델의 layers로 얻어지는 손실(loss)과 원본 이미지로 deep-dream 하는 과정에서, 손상된 이미지에서 과잉 해석되어 나오는 패턴으로 분석됨.
* 이러한 패턴이 부분적으로 생기는 이유는 이미지를 처음 해석할 때 임의의 위치에서 시작하기 때문이라고 추측함. 실제로 동일한 사진을 여러번 deep-dream을 시키면 다른 위치에서 유사한 패턴이 생기는 것에서 유추함.


### 2. 원본 이미지에서 규칙성이 보여지는 구역에서는 유사한 모양의 패턴이 반복적으로 등장함.

* 단적인 예로 일정한 규칙이 있는 이미지(벽돌)로 실험을 진행

<p align="center">
  <img src="https://i.esdrop.com/d/f/XDglyqtPeL/Bn8EZmGVxc.jpg" align="center" width="32%">
  <img src="https://i.esdrop.com/d/f/XDglyqtPeL/mkwz2CUOET.png" align="center" width="32%">
  <img src="https://i.esdrop.com/d/f/XDglyqtPeL/S9GmRqjtts.png" align="center" width="32%">
  <figcaption align="center">원본 이미지 / 1회 반복 / 10회 반복</figcaption>
</p>

* 이음새마다 기둥 모양의 패턴이 반복적으로 등장, 벽돌 부분에서는 동물, 하단부분에서는 탈 것의 모양 지속적으로 등장함.
* 비슷한 모양이 반복적으로 등장하는 부분은 원본 이미지의 영향이 있음.

### 3. 원본 이미지과 결과물을 비교했을 때, 많이 변형(손상)된 것 처럼 보이지만, 전체적으로 형태는 유지됨.

<p align="center">
  <img src="https://i.esdrop.com/d/f/XDglyqtPeL/qMTV1TZH7z.jpg" align="center" width="32%">
  <img src="https://i.esdrop.com/d/f/XDglyqtPeL/4Ozqm0AEOH.png" align="center" width="32%">
  <img src="https://i.esdrop.com/d/f/XDglyqtPeL/gqZyKbaxWf.png" align="center" width="32%">
  <figcaption align="center">원본 이미지 / 1회 반복 / 10회 반복</figcaption>
</p>

* 그룹으로 나눌 수 있는 구역에서 비슷한 패턴이 등장함.
* 단색의 하늘에서는 비슷한 모양의 패턴이 불규칙적인 위치에서 등장함.
* 건물 구역은 원본의 건물의 모양이 유지됨을 확인. 다만 더 과장되게 해석됨.
* 가게 구역은 탈 것의 모양이 지속적으로 등장함.
* 타일 바닥은 기존의 불규칙적인 패턴의 영향으로 다양한 패턴이 종합적으로 등장함.

# Installation

* ## 준비사항
	* [PyTorch](https://pytorch.org/) 다운로드
		*	**NVIDIA GPU** 사용을 권장합니다
	
	*  pre-training model  다운로드


		` python models/download_models.py -models all `
		

* ## 사용


	* 기본 
	
	
	 `python neural_dream.py -content_image <image_path/image.jpg> `


 	* Result에 사용된 코드
	
	
		`python neural_dream.py -content_image konkuk.jpg -image_size 1024 -output_image learninggif.png -create_gif -num_iterations 10 `
	
	
		`python neural_dream.py -content_image konkuk.jpg -learning_rate 0.5 -image_size 1024 -output_image learning5.png` 


		`python neural_dream.py -content_image konkuk.jpg -learning_rate 1.5 -image_size 1024 -output_image learning15.png` 


		`python neural_dream.py -content_image konkuk.jpg -learning_rate 3 -image_size 1024 -output_image learning30.png ` <br>


* ## 옵션 [[출처]](https://github.com/ProGamerGov/neural-dream#usage)

* 정정사항
	* -model_file 옵션의 디폴트 값이 VGG-19모델로 표기되어 있으나, 해당 모델은 구동이 안됨을 확인했습니다. 
	* 따라서 bvlc_googlenet.pth 이 디폴트 모델로 설정되어 내용과 다름을 알려드립니다.

	**Options**:


-   `-image_size`: Maximum side length (in pixels) of the generated image. Default is 512.
-   `-gpu`: Zero-indexed ID of the GPU to use; for CPU mode set  `-gpu`  to  `c`.

**Optimization options**:

-   `-dream_weight`: How much to weight DeepDream. Default is  `1e3`.
-   `-tv_weight`: Weight of total-variation (TV) regularization; this helps to smooth the image. Default is set to  `0`  to disable total-variation (TV) regularization.
-   `-l2_weight`: Weight of latent state regularization. Default is set to  `0`  to disable latent state regularization.
-   `-num_iterations`: Default is  `10`.
-   `-init`: Method for generating the generated image; one of  `random`  or  `image`. Default is  `image`  which initializes with the content image;  `random`  uses random noise to initialize the input image.
-   `-jitter`: Apply jitter to image. Default is  `32`. Set to  `0`  to disable jitter.
-   `-layer_sigma`: Apply gaussian blur to image. Default is set to  `0`  to disable the gaussian blur layer.
-   `-optimizer`: The optimization algorithm to use; either  `lbfgs`  or  `adam`; default is  `adam`. Adam tends to perform the best for DeepDream. L-BFGS tends to give worse results and it uses more memory; when using L-BFGS you will probably need to play with other parameters to get good results, especially the learning rate.
-   `-learning_rate`: Learning rate to use with the ADAM and L-BFGS optimizers. Default is  `1.5`. On other DeepDream projects this parameter is commonly called 'step size'.
-   `-normalize_weights`: If this flag is present, dream weights will be divided by the number of channels for each layer. Idea from  [PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer).
-   `-loss_mode`: The DeepDream loss mode;  `bce`,  `mse`,  `mean`,  `norm`, or  `l2`; default is  `l2`.

**Output options**:

-   `-output_image`: Name of the output image. Default is  `out.png`.
-   `-output_start_num`: The number to start output image names at. Default is set to  `1`.
-   `-leading_zeros`: The number of leading zeros to use for output image names. Default is set to  `0`  to disable leading zeros.
-   `-print_iter`: Print progress every  `print_iter`  iterations. Set to  `0`  to disable printing.
-   `-print_octave_iter`: Print octave progress every  `print_octave_iter`  iterations. Default is set to  `0`  to disable printing. If tiling is enabled, then octave progress will be printed every  `print_octave_iter`  octaves.
-   `-save_iter`: Save the image every  `save_iter`  iterations. Set to  `0`  to disable saving intermediate results.
-   `-save_octave_iter`: Save the image every  `save_octave_iter`  iterations. Default is set to  `0`  to disable saving intermediate results. If tiling is enabled, then octaves will be saved every  `save_octave_iter`  octaves.

**Layer options**:

-   `-dream_layers`: Comma-separated list of layer names to use for DeepDream reconstruction.

**Channel options:**

-   `-channels`: Comma-separated list of channels to use for DeepDream. If  `-channel_mode`  is set to a value other than  `all`  or  `ignore`, only the first value in the list will be used.
-   `-channel_mode`: The DeepDream channel selection mode;  `all`,  `strong`,  `avg`,  `weak`, or  `ignore`; default is  `all`. The  `strong`  option will select the strongest channels, while  `weak`  will do the same with the weakest channels. The  `avg`  option will select the most average channels instead of the strongest or weakest. The number of channels selected by  `strong`,  `avg`, or  `weak`  is based on the first value for the  `-channels`  parameter. The  `ignore`  option will omit any specified channels.
-   `-channel_capture`: How often to select channels based on activation strength; either  `once`  or  `octave_iter`; default is  `once`. The  `once`  option will select channels once at the start, while the  `octave_iter`  will select potentially new channels every octave iteration. This parameter only comes into play if  `-channel_mode`  is not set to  `all`  or  `ignore`.

**Octave options:**

-   `-num_octaves`: Number of octaves per iteration. Default is  `4`.
-   `-octave_scale`: Value for resizing the image by. Default is  `0.6`.
-   `-octave_iter`: Number of iterations per octave. Default is  `50`. On other DeepDream projects this parameter is commonly called 'steps'.
-   `-octave_mode`: The octave size calculation mode;  `normal`,  `advanced`,  `manual_max`,  `manual_min`, or  `manual`. Default is  `normal`. If set to  `manual_max`  or  `manual_min`, then  `-octave_scale`  takes a comma separated list of image sizes for the largest or smallest image dimension for  `num_octaves`  minus 1 octaves. If set  `manual`  then  `-octave_scale`  takes a comma separated list of image size pairs for  `num_octaves`  minus 1 octaves, in the form of  `<Height>,<Width>`.

**Laplacian Pyramid options:**

-   `-lap_scale`: The number of layers in a layer's laplacian pyramid. Default is set to  `0`  to disable laplacian pyramids.
-   `-sigma`: The strength of gaussian blur to use in laplacian pyramids. Default is  `1`. By default, unless a second sigma value is provided with a comma to separate it from the first, the high gaussian layers will use sigma  `sigma`  *  `lap_scale`.

**Zoom options:**

-   `-zoom`: The amount to zoom in on the image.
-   `-zoom_mode`: Whether to read the zoom value as a percentage or pixel value; one of  `percentage`  or  `pixel`. Default is  `percentage`.

**FFT options:**

-   `-use_fft`: Whether to enable Fast Fourier transform (FFT) decorrelation.
-   `-fft_block`: The size of your FFT frequency filtering block. Default is  `25`.

**Tiling options:**

-   `-tile_size`: The desired tile size to use. Default is set to  `0`  to disable tiling.
-   `-overlap_percent`: The percentage of overlap to use for the tiles. Default is  `50`.
-   `-print_tile`: Print the current tile being processed every  `print_tile`  tiles without any other information. Default is set to  `0`  to disable printing.
-   `-print_tile_iter`: Print tile progress every  `print_tile_iter`  iterations. Default is set to  `0`  to disable printing.
-   `-image_capture_size`: The image size to use for the initial full image capture and optional  `-classify`  parameter. Default is set to  `512`. Set to  `0`  disable it and  `image_size`  is used instead.

**GIF options:**

-   `-create_gif`: Whether to create a GIF from the output images after all iterations have been completed.
-   `-frame_duration`: The duration for each GIF frame in milliseconds. Default is  `100`.

**Help options:**

-   `-print_layers`: Pass this flag to print the names of all usable layers for the selected model.
-   `-print_channels`: Pass this flag to print all the selected channels.

**Other options**:

-   `-original_colors`: If you set this to  `1`, then the output image will keep the colors of the content image.
-   `-model_file`: Path to the  `.pth`  file for the VGG Caffe model. Default is the original VGG-19 model; you can also try the original VGG-16 model.
-   `-model_type`: Whether the model was trained using Caffe, PyTorch, or Keras preprocessing;  `caffe`,  `pytorch`,  `keras`, or  `auto`; default is  `auto`.
-   `-model_mean`: A comma separated list of 3 numbers for the model's mean; default is  `auto`.
-   `-pooling`: The type of pooling layers to use for VGG and NIN models; one of  `max`  or  `avg`. Default is  `max`. VGG models seem to create better results with average pooling.
-   `-seed`: An integer value that you can specify for repeatable results. By default this value is random for each run.
-   `-multidevice_strategy`: A comma-separated list of layer indices at which to split the network when using multiple devices. See  [Multi-GPU scaling](https://github.com/ProGamerGov/neural-dream#multi-gpu-scaling)  for more details. Currently this feature only works for VGG and NIN models.
-   `-backend`:  `nn`,  `cudnn`,  `openmp`, or  `mkl`. Default is  `nn`.  `mkl`  requires Intel's MKL backend.
-   `-cudnn_autotune`: When using the cuDNN backend, pass this flag to use the built-in cuDNN autotuner to select the best convolution algorithms for your architecture. This will make the first iteration a bit slower and can take a bit more memory, but may significantly speed up the cuDNN backend.
-   `-clamp`: If this flag is enabled, every iteration will clamp the output image so that it is within the model's input range.
-   `-adjust_contrast`: A value between  `0`  and  `100.0`  for altering the image's contrast (ex:  `99.98`). Default is set to 0 to disable contrast adjustments.
-   `-label_file`: Path to the  `.txt`  category list file for classification and channel selection.
-   `-random_transforms`: Whether to use random transforms on the image; either  `none`,  `rotate`,  `flip`, or  `all`; default is  `none`.
-   `-classify`: Display what the model thinks an image contains. Integer for the number of choices ranked by how likely each is.


* ## 에러 발생
	* 해당 소스코드 실행 시, 발생했던 에러 해결을 기술


* ### case 1 : AssertionError: Torch not compiled with CUDA enabled
	* 해당 코드로 해결 가능
	* 소스 코드 자체가 3년 전 마지막으로 업데이트이기에 최신 버전이 아니어도 구동 가능
	
	
	`pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
	
	
* ### case 2 : RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
	 * NVIDIA GPU 설치된 머신에서 실행 권장
	 * 본인은 노트북에서 실행이 안됨을 깨닫고, desktop에서 실행

# Presentation

[Click and watch youtube video](https://youtu.be/1RFnHQLftmU)