# DMCRID: Deep Nets for Multi-Camera Person Re-Identification

An implementation of the person re-identification

## Getting started

## Model Architecture

### Models

1. ResNet-50

2. PCBnet (Part-based Convolutional Baseline)

## Prerequisites

1. Matplotlib
2. Pytorch: torch and torchvision
3. Pillow
4. Scipy

**Installation**

Install the required packages by running the following command:

```bash
$ pip install -r requirements.txt
```

## Datasets

- [Market-1501]()

## Usage

### Part 1. Data preparation
Create a directory to store re-Id datasets under this repository or you can use the directory `datasets/`.

> **Note**
- If you wanna store datasets in another directory, you need to specify `--data-dir path_to_your_data` when running the training code.
- Please follow the instructions below to prepare datasets.
- If you find any errors/bugs, please feedback in the `Issues` section.
- In the following, I assume that the path to the dataset directory is `datasets/`.

#### Market-1501:
1. Download the dataset to `datasets/` from [here](http://www.liangzheng.org/Project/project_reid.html).
2. Extract the file and rename it to `Market-1501`. The data structure should look like:
```
├── Market-1501/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* We do not use it 
│   ├── gt_query/                   /* Files for multiple query testing 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
```

3. Run the following script to transform the dataset, replacing the paths with yours by opening and editing the file `prepare_data.py` and then change the `root_path = './datasets/Market-1501'` to your dataset path.  
```bash
$ python prepare_data.py
```

### Part 2. Training

### Part 3. Evaluation

### Part 4. Demo

## Examples

## TODO

- [x] Data preparation
- [x] Create the model
- [x] Train the defined model
- [x] Running inference

## References

- https://pytorch.org/
- [A Practical Guide to Person Re-Identification Using AlignedReID](https://medium.com/@niruhan/a-practical-guide-to-person-re-identification-using-alignedreid-7683222da644)
- [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/pdf/1711.09349.pdf)
- [Person Re-Identification by Deep Learning Multi-Scale Representations](http://www.eecs.qmul.ac.uk/~xiatian/papers/ChenEtAl_ICCV2017WS_CHI.pdf)