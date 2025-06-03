该项目在Flare-Free Vision的基础上进行修改，训练命令与评估命令未变，脚本内容与代码内容发生改变。
注意运行时显示int 转变 torch为正常现象。

## Table of Contents
- [Flare-Free Vision: Empowering Uformer with Depth Insights](#flare-free-vision-empowering-uformer-with-depth-insights)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Training](#training)
    - [Training Using Single GPU](#training-using-single-gpu)
    - [Training Using Multiple GPUs](#training-using-multiple-gpus)
  - [Testing](#testing)
    - [Saved Checkpoint](#saved-checkpoint)
    - [Inference](#inference)
    - [Evaluation](#evaluation)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)
## Installation
1. Clone the repository
```bash
git clone https://github.com/sfffffffffffff/DIP.git
```

2. Install the requirements
```bash
cd Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights
pip install -r requirements.txt
```

3. Install BasicSR
```bash
python setup.py develop
```

4. Install pretrained [Dense Vision Transformer](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view) for Depth Prediction and put it in `DPT` folder.
5.下载60k ir训练，模型与损失函数改进的预训练模型为：

## Dataset

The dataset used in training and testing is uploaded on [Google Drive](https://drive.google.com/file/d/1rQ2ZG3HHoBOogYw_qnH3SgLlNlsQtPST/view?usp=sharing). To avoid multiple errors, follow the following structure:

```
dataset (folder)
├── README.md
├── Flare7Kpp
│   ├── Flare7K
|   |── |── Scattering_Flare
|   |── |── |── Compound_Flare
|   |── |── |── Light_source
|   |── Flare-R
|   |── |── Compound_Flare
|   |── |── Light_source
|   |── test_data
|   |── |── real
|   |── |── |── gt
|   |── |── |── input
|   |── |── |── mask
|   |── |── synthetic
|   |── |── |── gt
|   |── |── |── input
|   |── |── |── mask
|   |── val
|   |── |── gt
|   |── |── input
|   |── |── mask
|── Flickr24K
```

To unzip the dataset, run the following command:

```bash
unzip dataset.zip -d dataset
```

## Training

### Training Using Single GPU
To start training, you need to configure your training parameters in `options/uformer_flare7kpp_baseline_option.yml`. Then, run the following command:

```python
python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml
```

**Note:** you can start autmotaically from a checkpoint by adding `--auto_resume` to the command above.
```python
python basicsr/train.py -opt options/uformer_flare7kpp_baseline_option.yml --auto_resume
```

### Training Using Multiple GPUs
To start training using multiple GPUs, you need to configure your training parameters in `options/uformer_flare7kpp_baseline_option.yml`. Then, run the following command:

```python
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 options/uformer_flare7kpp_baseline_option.yml
```

## Testing

### Saved Checkpoint
We have uploaded our trained model on [Google Drive](https://drive.google.com/file/d/13SWmwJVaRn6tUuJX2fljbrrxQSr9xVF6/view?usp=sharing). To use it, you can download it and put it in `experiments/flare7kpp/pretrained.pth`.

### Inference
To start inference on test dataset, you can run the following command:

```python
python basicsr/inference.py --input dataset/Flare7Kpp/test_data/real/input/ --output result/real/pretrained/ --model_path experiments/flare7kpp/pretrained.pth --flare7kpp
```

### Evaluation
To evaluate the performance of the model using PSNR, SSIM, LPIPS, Glare PSNR, and Streak PSNR, you can run the following command:

```python
python evaluate.py --input result/real/pretrained/blend/ --gt dataset/Flare7Kpp/test_data/real/gt/ --mask dataset/Flare7Kpp/test_data/real/mask/
```

## License
This project is governed by the S-Lab License 1.0. If you intend to redistribute or utilize the code for non-commercial purposes, it is imperative to adhere to the terms outlined in this license

## Acknowledgement
This work borrows heavily from [Flare7K++: Mixing Synthetic and Real Datasets for Nighttime Flare Removal and Beyond](https://github.com/ykdai/Flare7K). We would like to thank the authors for their work.
