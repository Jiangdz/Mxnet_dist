#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function
import random, sys

import mxnet as mx
from mxnet import autograd, gluon, kv, nd
#from mxnet.gluon.model_zoo import vision
from mxnet.gluon.data.vision import transforms

import numpy as np
import time

from gluoncv.model_zoo import get_model

import logging
import os

logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
DATEFMT ="[%Y-%m-%d %H:%M:%S]"
FORMAT = "%(asctime)s %(thread)d %(message)s"
def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=output_dir+'.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

logging = init_log('mxnet_test')
_print = logging.info


# Create a distributed key-value store
store = kv.create('dist') # Note: you can control the sync and async here (https://mxnet.incubator.apache.org/api/python/kvstore/kvstore.html)


# Clasify the images into one of the 10 digits
num_outputs = 10

# 64 images in a batch
batch_size_per_gpu = 256  
# How many epochs to run the training
epochs = 2

# How many GPUs per machine
gpus_per_machine = 1 # Note: Configure the GPU number
# Effective batch size across all GPUs
batch_size = batch_size_per_gpu * gpus_per_machine

# Create the context (a list of all GPUs to be used for training)
ctx = [mx.gpu(i) for i in range(gpus_per_machine)] # Note: maybe you can configure gpu/cpu here, please check it
#ctx = mx.gpu(2)
# Convert to float 32
# Having channel as the first dimension makes computation more efficient. Hence the (2,0,1) transpose.
# Dividing by 255 normalizes the input between 0 and 1
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)




transform_train = transforms.Compose([
    # Randomly crop an area, and then resize it to be 32x32
    transforms.RandomResizedCrop(32),
    # Randomly flip the image horizontally
    transforms.RandomFlipLeftRight(),
    # Randomly jitter the brightness, contrast and saturation of the image
    transforms.RandomColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    # Randomly adding noise to the image
    transforms.RandomLighting(0.1),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation calculated across all images
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])




class SplitSampler(gluon.data.sampler.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the part with index `part_index`

    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Partition the data into multiple parts
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0):
        # Compute the length of each partition
        self.part_len = length // num_parts
        # Compute the start index for this partition
        self.start = self.part_len * part_index
        # Compute the end index for this partition
        self.end = self.start + self.part_len

    def __iter__(self):
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(range(self.start, self.end))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.part_len

# Load the training data
train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
                                      batch_size,
                                      sampler=SplitSampler(50000, store.num_workers, store.rank))

# Load the test data 
test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
                                     batch_size, shuffle=False)

# Use ResNet from model zoo

"""
classes = 10
model_name = "res"
if model_name == "res":
    net = vision.resnet34_v1(classes=classes)
elif model_name == "alex":
    net = vision.alexnet(classes=classes)
elif model_name == "squeeze":
    net = vision.squeezenet1_0(classes=classes)
elif model_name == "dense":
    net = vision.densenet161(classes=classes)
elif model_name == "cnn":
    net = gluon.nn.HybridSequential(prefix='')
    with net.name_scope():
        net.add(gluon.nn.Conv2D(64, kernel_size=11, strides=4, padding=2, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
        net.add(gluon.nn.Dense(1024, activation='relu'))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(classes))
"""

# Note : choose the model here
model_name = "resnet"
if model_name == "resnet":
    net_name = 'cifar_resnet110_v2'
    net = get_model(net_name, classes=10)
elif model_name == "densenet":
    from densenet import DenseNet
    net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
elif model_name == "cnn":
    net = gluon.nn.HybridSequential(prefix='')
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=5, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(0.25))
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=5, activation='relu'))
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(0.25))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(10))
elif model_name == "mlp":
    net = gluon.nn.HybridSequential(prefix='')
    with net.name_scope():
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(200, activation="relu"))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(10))

net.hybridize()
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

"""
# Use Adam optimizer. Ask trainer to use the distributer kv store.
if False : #del_name == "cnn" or model_name=="mlp":
    optimizer = 'sgd'
    optimizer_params = {'learning_rate': 0.01}
else:
    optimizer = "nag"
    optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}
lr_decay_epoch = [80, 160, np.inf]
"""
if model_name=="mlp":
    optimizer = "adam"
    optimizer_params = {'learning_rate': 0.001}
    decay_rate = 0.1
elif model_name=="cnn":
    optimizer = "adam"
    optimizer_params = {'learning_rate': 0.001}
    decay_rate = 0.1
elif model_name == "resnet":
    optimizer = "sgd"
    optimizer_params = {'learning_rate': 0.1, "momentum":0.9}
    decay_rate = 0.1
elif model_name == "densenet":
    optimizer = "sgd"
    optimizer_params = {'learning_rate': 0.1, "momentum":0.9}
    decay_rate = 0.1
lr_decay_epoch = [78, np.inf]



trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, kvstore=store)

# Evaluate accuracy of the given network using the given data
def evaluate_accuracy(data_iterator, net):

    acc = mx.metric.Accuracy()

    # Iterate through data and label
    for i, (data, label) in enumerate(data_iterator):

        # Get the data and label into the GPU
        data = data.as_in_context(ctx[0])
        label = label.as_in_context(ctx[0])

        # Get network's output which is a probability distribution
        # Apply argmax on the probability distribution to get network's classification.
        output = net(data)
        predictions = nd.argmax(output, axis=1)

        # Give network's prediction and the correct label to update the metric
        acc.update(preds=predictions, labels=label)

    # Return the accuracy
    return acc.get()[1]

# We'll use cross entropy loss since we are doing multiclass classification
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# Run one forward and backward pass on multiple GPUs
def forward_backward(net, data, label):
    #print("AAAA", time.time())
    # Ask autograd to remember the forward pass
    with autograd.record():
        # Compute the loss on all GPUs
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]

    # Run the backward pass (calculate gradients) on all GPUs
    #losses.backward()
    #print("BBBB", time.time())
    for l in losses:
        l.backward()
    #print("CCCC", time.time())
    #return np.sum(losses.asnumpy())

# Train a batch using multiple GPUs
def train_batch(batch, ctx, net, trainer):
    #print("0000", time.time())
    # Split and load data into multiple GPUs
    data = batch[0]
    data = gluon.utils.split_and_load(data, ctx)

    # Split and load label into multiple GPUs
    label = batch[1]
    label = gluon.utils.split_and_load(label, ctx)
    #print(data)
    #print(label)
    # Run the forward and backward pass
    #print("1111", time.time())
    forward_backward(net, data, label)
    #print(ll)
    # Update the parameters
    #print("2222", time.time())
    this_batch_size = batch[0].shape[0]
    trainer.step(this_batch_size)
    #print("3333", time.time())

start_time = time.time()
last_time = start_time
# Run as many epochs as required
for epoch in range(epochs):
    if epoch in lr_decay_epoch:
        trainer.set_learning_rate(trainer.learning_rate*decay_rate)
    # Iterate through batches and run training using multiple GPUs
    batch_num = 1
    for batch in train_data:

        # Train the batch using multiple GPUs
        train_batch(batch, ctx, net, trainer)

        batch_num += 1
    # test only in GPU 0 so we also show the train time 
    # print("Epoch %d: Train time %f" % (epoch, time.time()-last_time,))
    _print('Epoch {}: Train time {}'.format(epoch, time.time()-last_time))
    # Print test accuracy after every epoch
    test_accuracy = evaluate_accuracy(test_data, net)
    # print("Epoch %d: Test_acc %f Time_per %f Time_tatal %f" % (epoch, test_accuracy, time.time()-last_time, time.time()-start_time))
    _print('Epoch {}: Test_acc {} Time_per {} Time_tatal {}'.format(epoch, test_accuracy, time.time()-last_time, time.time()-start_time))
    last_time = time.time()
    sys.stdout.flush()
print("Finish")
