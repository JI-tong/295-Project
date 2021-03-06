# Smart Traffic Management Using Deep Learning
#                       - CMPE-295 Project

## Team member:  Ji Tong, Yishi Chen, Zening Deng.

### Tried 2 methods:
        * Tensorflow Object Detection API
        * Keras Retinanet

## Steps:

### a. preprocessing data
```
1. find get dataset: DATRAC dataset and SJSU street camera dataset.

2. labeling the dataset images, generate xml file.

3. transfer xml to csv.

4. (Tensorflow object detection API only) : generate TFrecords.
```

### b. training:

#### basic tensorflow object detection API:

```
1. download the ssd_mobilenet config file and made necessary changes (class numbers, types, data directory, etc).

2. go to tensorflow object detection api folder and do: 
        python3 train.py \
        --logtostderr   \
        --train_dir=output_directory  \
        --pipeline_config_path=path/to/config/file
```

#### keras retinanet:

```
1. create a class_map.csv file for class and its labels (e.g. car,1 ; van, 2)

2. go to keras retinanet do: 
        ./keras_retinanet/bin/train.py csv training_image.csv class_map.csv
```

### c. test and evaluation:

#### basic tensorflow object detection API:

```
1.modify config file 

2.go to tensorflow object detection api folder and do:
       python3 eval.py \
       --logtostderr \
       --pipeline_config_path=path/to/config/file
       --checkpoint_dir=path/to/checkpoint
       --eval_dir=path/to/save/eval
       
3.visualize evaluation:
       tensorboard --logdir=path/to/save/eval
```

#### keras retinanet:

```
1. convert model, go to keras-retinanet bin folder, do:
        python3 convert_model.py trained_model.h5 converted_model.h5

2. evaluate, go to keras-retinanet folder, find and do:
        python3 evaluate.py csv test_image.csv class_map.csv converted_model.h5  --save-path=path/to/save/images

```

### d. tracking:

#### for tracking, just do:  
```
python3 main.py video.mp4
```
#### explaination: 
#### basic tensorflow object detection API:
```
codes are all in the trackingOnTensorflowModel.

based on github https://github.com/srianant/kalman_filter_multi_object_tracking and made some modification

1. rewrite the predicted class type.

2. add cars id functionality, give each car an id, so we can know during a specific time period, how many cars has passed the crossroad, and use the coordinate changes associated with each id, to predict the speed of the car
```
#### Keras retinanet:
```
codes are all in the trackingOnKerasModel.

similar to the previous one, but rewrite the whole detector.py, since both model has different format.
Keras is using h5 format, however tensorflow is using pb format.

```
### e. anticipate crashing:

```
codes are all in anticipation folder
```
#### for anticipating, just do:

```
go to anticipation folder, do:
       python3 anticipating.py --model path/to/model
```
