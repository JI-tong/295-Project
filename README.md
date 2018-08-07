# Smart Traffic Management Using Deep Learning
#                       - CMPE-295 Project

## Team member:  Ji Tong, Yishi Chen, Zening Deng.

### Tried 2 models:
        * basic tensorflow object detection API
        * retina net

#### basic tensorflow object detection API:
first go to mymodels directory, run below code to generate the tfrecords:
```
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
```
then:
1) do:
suppose '~/Path' is the path to access the tensorflow obejct detection API
```
export PYTHONPATH=$PYTHONPATH: ~/Path/models/research: ~/Path/models/research/slim
```

2) do:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
3) do:
export an object detection model for inference.
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-1228 --output_directory inference_graph

```
4) do:
```
./Object_detection_xxxx.py based on the test set format
```

#### retina net:

1) run:
```
class_map_to_csv.py && xml_to_csv.py
```

2) go to the retina net directory, do:

```
./keras_retinanet/bin/train.py csv path/to/the/csv/file/train_labels.csv path/to/the/csv/file/class_map.csv 
```

