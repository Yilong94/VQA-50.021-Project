# Visual Question Answer Task for 50.021 Artificial Intelligence project

This is an implementation of the paper [Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering in PyTorch](https://arxiv.org/abs/1704.03162). Note that instead of using attention in our model, we used liner feedfoward network to fill the gap, hence the model does not achieve the same reported result as in the paper. We also created a simple GUI that loads an image and a question, in which the model will generate an answer.

## Getting Started

1. Open config.py in a text editor and set the paths with respect to local machine directories
2. For preprocessing and training settings, set parameters with respect to your machines
specifications. We ran on a GTX Titan XP.
3. Ensure your machine has more than 100gb worth of storage for the preprocessed data

**Pre-processing of images, questions and answers:**
(Allows re-usage of data features to speed up training, as opposed to generating same data features every epoch)
1. Preprocess images using pre-trained ResNet-152
- Run command line: python3 preprocess-images.py
2. Preprocess questions and answers by converting into JSON format
- Run command line: python3 preprocess-vocab.py

**Training of model:**
1. Run command line: python3 train.py
2. To view train progress, from another command line: python3 view-log.py logs/(model_name).pth

**Running the GUI:**
1. Ensure the following files are in the correct directory:
- image_dir = config.val_path
- image_features_dir = config.preprocessed_path
- question_dir = "vqa/v2_OpenEnded_mscoco_val2014_questions.json"
- vocab_dir = "vocab.json"
- checkpoint = "model.pth"
2. Ensure the saved model is in the same directory as “gui.py”.
3. Rename saved model to “model.pth”
4. Run command line: python3 gui.py

### Prerequisites

* Environment: Python 3.6
* torchvision==0.1.9
* torch==0.4.0
* h5py==2.7.0
* tqdm==4.22.0

## Authors

* **Burindy** - *implement and edit Resnet classifier*
* **Paul Tan** - *integration of LSTM model, average pooling into final model, preprocessing images, setting up
machine environment and training on cv lab computer/AMI, assisted in GUI data loading component*
* **Tan Jia Wen** - *implementation of dataset and dataloader class, preprocessing of questions and answers,
assisted with the training of the model, selection of validation images*
* **Tan Yi Long** - *implementation of LSTM model, GUI design and implementation*

## Acknowledgments

* Code was heavily inspired from [Cyanogenoid](https://github.com/Cyanogenoid)