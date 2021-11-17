# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.data.experimental import AUTOTUNE
from functools import partial
import numpy as np
import os
from glob import glob
import argparse
from functools import reduce
from icecream import ic

print(f"TF version: {tf.version.VERSION}")


def load_image(image_path, image_label):
    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (128, 128))
   
    return (image, image_label)

def get_label(file_path, class_names):
    class_name = tf.strings.split(file_path, "/")[-2]
    label = tf.argmax(tf.cast(class_name == class_names, tf.int32))
    
    return file_path, label

def augment_image(image, label, augmenter):
    
    image = augmenter(image)

    return (image, label)


def make_ds(image_paths, class_names, batch_size=32, shuffle=True, augment=True, augmenter=None):

    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    get_labels = partial(get_label, class_names=class_names)
    ds = ds.map(get_labels)
    
    if shuffle:
        ic()
        ds = ds.shuffle(len(image_paths), seed=1234)
    
    ds = (ds.map(load_image, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(batch_size)
         )
    
    if augment:
        ic()
        image_augmentation = partial(augment_image, augmenter=augmenter)
        ds = ds.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    
    ds = ds.prefetch(AUTOTUNE)
          
    return ds


def build_model(base_model, num_classes, dropout, num_fully_connected_layers, num_unit_per_layer):
    
    x = Flatten()(base_model.output)
    fc_layers = reduce(lambda x,y: y(x), [Dense(units=num_unit_per_layer, activation="relu") for _ in range(num_fully_connected_layers)], Flatten()(base_model.output))
    x = BatchNormalization(axis=-1)(fc_layers)
    x = Dropout(rate=dropout)(x)
    x = Dense(units=num_classes, activation="softmax")(x)
    
    return tf.keras.Model(inputs=base_model.input, outputs=x)


def main(args):
    # Create data generators for feeding training and evaluation based on data provided to us
    # by the SageMaker TensorFlow container
    
    if args.debug:
        ic.enable()
        ic.configureOutput(prefix='debug | ')
        ic(args)
    else:
        ic.disable()
    
    class_names = os.listdir(args.train)
    ic(len(class_names))
    
    train_files = glob(f"{args.train}/*/*")
    ic(len(train_files))
    
    test_files = glob(f"{args.test}/*/*")
    ic(len(test_files))
    
    # This is a keras model that will be used for data augmentation
    augmenter = tf.keras.Sequential([
            preprocessing.RandomFlip("horizontal_and_vertical"),
            preprocessing.RandomZoom(
                height_factor=(-0.05, -0.15),
                width_factor=(-0.05, -0.15)),
            preprocessing.RandomRotation(0.3)
            ])
    
    train_ds = make_ds(train_files, class_names, augment=True, augmenter=augmenter, batch_size=args.batch_size)
    test_ds = make_ds(train_files, class_names, augment=False)
    
    base_model = MobileNetV2(weights='imagenet',
                       include_top=False,
                       input_shape=(128, 128, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model = build_model(base_model, len(class_names), args.dropout, args.num_fully_connected_layers, args.num_unit_per_layer)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  metrics=['accuracy']
                 )
    
    s3_job_dir = os.path.dirname(os.environ.get("SM_MODULE_DIR")).replace("/source","")
    tb_log_dir = f"{s3_job_dir}/tboard_logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir, 
                                                          write_images=True,
                                                          histogram_freq=1,
                                                          update_freq="epoch")
    
    history = model.fit(x=train_ds, validation_data=test_ds, epochs=args.epochs, callbacks=[tensorboard_callback])
    model.save(os.path.join(os.environ.get("SM_MODEL_DIR"), "001"))

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_fully_connected_layers', type=int, default=1)
    parser.add_argument('--num_unit_per_layer', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()
    print('args: {}'.format(args))
    
    main(args)