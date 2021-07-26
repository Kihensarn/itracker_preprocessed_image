# itracker_high_resolution_image
## Introduction
This project is also to learn about the process of gaze estimation. Among this project, the model is based on itracker and the dataset is ETH-Xgaze dataset. Meanwhile, the preprocessed data comes from [VIPL-TAL-GAZE/GAZE2021](https://github.com/VIPL-TAL-GAZE/GAZE2021), which are more precise than the data preprocessed by myself.
## Requirements
* **Python** == 3.8
* **Torch** == 1.7.0+cu101
* **Visdom** == 0.1.8.9  

More details in [requirements.txt](itracker_preprocessed_image/requirements.txt) file.
## File Structure
* **itracker_preprocessed_image** 
    * **data**   
    This directory contains the files to process the dataset.
    * **model**  
    This directory contains various backbone networks, such as resnet, deit, mobilenet and so on. Meanwhile, it also contains the model we use to estimate the 3D gaze vector.
    * **options**   
    This directory contians several config file, which set hyperparameters, data path, model path and so on.
    * **utils**  
    This directory contains some useful files such as loss.py, logger.py, module.py.These file can be used to evaluate the results and record the log and so on.
    * **scripts**  
    This directory contains some files to train or test the model.
    * **results**  
    This directory stores the checkpoints and log file.
* **dataset**  
This directory stores the preprocessed data. Detailed file structure is showed below.
```
├── dataset		
│   ├── train
│   │   ├── subject0000
│   │   │   ├── face
│   │   │   |   ├── 000000.jpg
│   │   │   |   ├── 000001.jpg
│   │   │   |   ├── ...  
│   │   │   ├── left_eye
│   │   │   |   ├── 000000.jpg
│   │   │   |   ├── 000001.jpg
│   │   │   |   ├── ... 
│   │   │   ├── right_eye
│   │   │   |   ├── 000000.jpg
│   │   │   |   ├── 000001.jpg
│   │   │   |   ├── ... 
│   │   ├── ...
│   ├── test
│   │   ├── subject0001
│   │   │   ├── face 
│   │   │   ├── left_eye
│   │   │   ├── right_eye
│   │   ├── ...
│   ├── val
│   │   ├── subject0003
│   │   │   ├── face 
│   │   │   ├── left_eye
│   │   │   ├── right_eye
│   │   ├── ...
```
## Results
 model  | train_error  | val_error  | test_error  | test_error_std
 ---- | ----- | ------ | ------ | ------  
 itracker_preprocessed_image  | 1.4662 | 5.1491 | 5.3161 | 5.2669
## Usage
Prepare the ETH-Xgaze dataset
```
python prepareXgaze.py --dataset_path [source dataset path] --outer_dataset_path [the directory to store preprocessed data]
```
Train itracker model
```
python xgaze_main.py --data_path [preprocessed dataset path] --train True --load False
```
Test itracker model
```
python xgaze_main.py --data_path [preprocessed dataset path] --train False --load True
```
## Citation
For itracker model, please cite:
```
@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```
For ETH-Xgaze dataset, please cite:
```
@inproceedings{Zhang2020ETHXGaze,
  author    = {Xucong Zhang and Seonwook Park and Thabo Beeler and Derek Bradley and Siyu Tang and Otmar Hilliges},
  title     = {ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation},
  year      = {2020},
  booktitle = {European Conference on Computer Vision (ECCV)}
}
```
