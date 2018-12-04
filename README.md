# pytorch-basic-module

A wrapped basic model class for pytorch models. You can Inherit it to make your model easier to use. It contains methods such as load, save, multi-thread save, parallel distribution, train, validate, predict and so on. 

## Functions

### load

Load the existing model.
:param model_type: temp model or best model.
:param map_location: your working environment.
    For loading the model file dumped from gpu or cpu.
:return: None.

### save

Save the current model.
:param epoch:The current epoch (sum up). This will be together saved to file, aimed to keep tensorboard curve a continuous line when you train the net several times.
:param loss:Current loss.
:param name:The name of your saving file.
:return:None

### mt_save

Save the model with a new thread. You can use this method in stead of self. save to save your model while not interrupting the training process, since saving big file is a time-consuming task. Also, this method will automatically record your best model and make a copy of it.
:param epoch: Current loss.
:param loss:
:return: None

### _get_optimizer

Get your optimizer by parsing your opts.
:return:Optimizer.

### to_multi

If you have multiple GPUs and you want to use them at the same time, you should call this method before training to send your model and data to multiple GPUs.
:return: None

### validate

Validate your model.
:param test_loader: A DataLoader class instance, which includes your validation data.
:return: test loss and test accuracy.

### predict

Make prediction based on your trained model. Please make sure you have trained your model or load the previous model from file.
:param test_loader: A DataLoader class instance, which includes your test data.
:return: Prediction made.

### fit

Training process. You can use this function to train your model. All configurations are defined and can be modified in config.py.
:param train_loader: A DataLoader class instance, which includes your train data.
:param test_loader: A DataLoader class instance, which includes your test data.
:return: None.
