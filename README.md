# basic_cv
a basic code repository for basic task in CV(classification,detection,segmentation,tracking)

## classification
The root dir of classification is assumed as ```basic_cv/classification/```. 
### generate dataset
All data should be organized under ```./datasets/``` dir as follow:
```
# prepare your dataset in the following structure:
./datasets/train/*.jpg	# train dataset
./datasets/test/*.jpg	# test dataset
./datasets/train.txt	# txt for train dataset  
./datasets/test.txt		# txt for test dataset
```
Label and image name will be saved in ```train.txt``` and ```test.txt``` as follow: 
```
image1.jpg 1	# image_name,label
image2.jpg 3
image3.jpg 0
...
```
If your data has already be organized as follow:
```
./datasets/0/*.jpg	# data with label 0
./datasets/1/*.jpg	# data with label 1
...
```
Then, use torchvision.datasets.ImageFolder as it is useful for this kind of dataset structure.
If you want to get a larger dataset by data augmentation,use tools in ```./datasets/data_enhance.py```.

### train
```
# train your classification model(with config)
python train.py --cfg config.yaml
```
The model will be saved in dir  ```./output```.
If your want to design your own model, get your own model in ```./models```, and use it in ```train.py```.
### predict
```
# test your classification model(with config)
python demo.py --cfg config.yaml
```
## detection

## segmentation

## tracking
