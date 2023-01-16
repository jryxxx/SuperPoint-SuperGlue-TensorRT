# SuperPoint SuperGlue TensorRT
SuperPoint and SuperGlue with TensorRT.

## Demo
<img src="image/superpoint_superglue_tensorrt.gif" width = "640" height = "240"  alt="match_image" border="10" />

* This demo was tested on the Quadro P620 GPU.

## Baseline

| Image Size: 320 x 240  | RTX3080 | Quadro P620 | 
|:----------------------:|:-------:|:-----------:|
| SuperPoint (250 points)|         | 13.61 MS    | 
| SuperPoint (257 points)|         | 13.32 MS    | 
| SuperGlue (256 dims)   |         | 58.83 MS    |

## Docker(Recommand)
```bash
docker pull yuefan2022/tensorrt-ubuntu20.04-cuda11.6:latest
docker run -it --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --privileged --runtime nvidia --gpus all --volume ${PWD}:/workspace --workdir /workspace --name tensorrt yuefan2022/tensorrt-ubuntu20.04-cuda11.6:latest /bin/bash
```

## Environment Required
* CUDA==11.6
* TensorRT==8.4.1.5
* OpenCV>=4.0
* EIGEN
* yaml-cpp

## Convert Model(Optional)
The converted model is already provided in the [weights](./weights) folder, if you are using the pretrained model officially provided by [SuperPoint and SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), you do not need to go through this step.
```bash
python convert2onnx/convert_superpoint_to_onnx.py --weight_file superpoint_pth_file_path --output_dir superpoint_onnx_file_dir
python convert2onnx/convert_superglue_to_onnx.py --weight_file superglue_pth_file_path --output_dir superglue_onnx_file_dir
```

## Build and Run
```bash
git clone https://github.com/yuefanhao/SuperPointSuperGlueAcceleration.git
cd SuperPointSuperGlueAcceleration
mkdir build
cd build
cmake ..
make
# test on image pairs 100 times, the output image will be saved in the build dir
./superpointglue_image  ../config/config.yaml ../weights/ ${PWD}/../image/image0.png ${PWD}/../image/image1.png
# test on the folder with image sequence, output images will be saved in the param assigned dir
./superpointglue_sequence  ../config/config.yaml ../weights/ ${PWD}/../image/freiburg_sequence/ ${PWD}/../image/freiburg_sequence/match_images/
```
The default image size param is 320x240, if you need to modify the image size in the config file, you should delete the old .engine file in the weights dir.
