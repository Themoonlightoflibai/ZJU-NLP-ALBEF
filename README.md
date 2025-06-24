# ZJU-NLP-ALBEF
Course project of ZJU NLP class

## Data Process
### Tiny Image Dataset
Due to the limited condition, we can only use a device with 32GB memory, so we have to decline the number of the data in the dataset

The original data is VQA dataset, which contains 83k images for training, 42k images for validation and 81k images for testing. We only use the data in Val dataset for training, validation and testing.

The script `tinydataset.py` is used to generate tiny dataset from val dataset. If you want to design your own dataset, feel free to change the `json` file name and the address of dataset directory.

The script `tinyMindeSporeGenerator.py` is used to create MindRecord file from tiny dataset.
### Image Dataset
The original dataset is VQA dataset, you can use the following code to generate MindRecord file from the dataset

If you have enough GPUs, feel free to try it.
### Annotations
The annotation files is from the official ALBEF repo. please dowload them.


