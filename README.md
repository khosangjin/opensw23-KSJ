
# opensw23-KSJ

# Team Introduction
* 이름 : khosangjin 
 * ID :  202011248

# Topic Introduction
* ## 딥 드림 (Deep-Dream, Neural-dream)
	* 딥 드림(Deep-Dream)은 합성곱-신경망(Convolutional neural network, CNN)이 학습한 패턴을 시각화(Visualize)한 알고리즘입니다.
	
	* 입력된 이미지에 의도적으로 패턴을 과잉 해석함으로써, hallucinate하고 몽환적인 느낌의 사진을 출력합니다.

# Results
![원본](https://i.esdrop.com/d/f/XDglyqtPeL/4kJdlMe3t1.jpg)
![가중치 0.5 적용](https://https://i.esdrop.com/d/f/XDglyqtPeL/Sz5ZQtKTDl.png)
![가중치 1.5 적용(default)](https://https://i.esdrop.com/d/f/XDglyqtPeL/qheN0WnrGC.png)
![가중치 3.0 적용](https://i.esdrop.com/d/f/XDglyqtPeL/B6WHYGEB7V.png)


# Analysis/Visualization

# Installation
* ## 준비사항
	* [PyTorch](https://pytorch.org/) 다운로드
	
	*  pre-training model  다운로드
		` python models/download_models.py -models all `

* ## 사용
	* 기본 
	 `python neural_dream.py -content_image <image.jpg> `
	 * 옵션 +
	` python neural_dream.py -content_image examples/inputs/konkuk.jpg -output_image path_nin_cudnn.png -gpu 0 -backend cudnn -num_iterations 10 -dream_weight 10 -image_size 1024 -optimizer adam -learning_rate 3`

* ## 옵션
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
