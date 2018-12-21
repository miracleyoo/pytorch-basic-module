# pytorch-basic-module

A wrapped basic model class for pytorch models. You can Inherit it to make your model easier to use. It contains methods such as load, save, multi-thread save, parallel distribution, train, validate, predict and so on. 

## Usage

Generally, you will inherit **nn.module** class when you define your own net, when you use this library, what you have to change is to inherit **BasicModule** instead. Also, you need to put the **config.py** file and **BasicModule.py** file into your main directory, and import them of course.

### Example

`YourModule.py`

```python
from .BasicModule import *
class YourModule(BasicModule):
    def __init__(self, opt, input_size=224, width_mult=1.):
        super(YourModule, self).__init__(opt=opt)
        ......
```

`main.py`

```python
# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

from config import Config
from models.MobileNetV2 import *
from utils.utils import *


def main():
    # Initializing Configs
    folder_init(opt)

    # Initialize model
    try:
        if opt.MODEL_NAME == 'MobileNetV2':
            net = MobileNetV2(opt)
    except KeyError('Your model is not found.'):
        exit(0)


    if opt.START_PREDICT: # Prediction Part
        # Load saved model
        net.load(model_type="best_model.dat")
        # Send the model to its device(maybe multiple GPU)
        net = prep_net(net)
        # Load Data
        _, val_loader = load_regular_data(opt, net, val_loader_type=ImageFolder)
        # Start validating
        predict(net, val_loader)
    else: # Training Part
        # Load saved model
        if opt.LOAD_SAVED_MOD:
            net.load()
        # Send the model to its device(maybe multiple GPU)
        net = prep_net(net)
        # Load Data
        if net.opt.DATALOADER_TYPE == "ImageFolder":
            train_loader, val_loader = load_regular_data(opt, net, train_loader_type=ImageFolder)
            log("All datasets are generated successfully.")
        else:
            raise KeyError("Your DATALOADER_TYPE doesn't exist!")
        # Start training
        fit(net, train_loader, val_loader)

if __name__ == '__main__':
    # Options and Argparses
    opt = Config()
    parser = argparse.ArgumentParser(description='Training')
    pros = [name for name in dir(opt) if not name.startswith('_')]
    abvs = ['-' + ''.join([j[:2] for j in i.split('_')]).lower()[:3] if len(i.split('_')) > 1 else
            '-' + i.split('_')[0][:3].lower() for i in pros]
    types = [type(getattr(opt, name)) for name in pros]
    with open('./reference/help_file.pkl', 'rb') as f:
        help_file = pickle.load(f)
    for i, abv in enumerate(abvs):
        if types[i] == bool:
            parser.add_argument(abv, '--' + pros[i], type=str2bool, help=help_file[pros[i]])
        else:
            parser.add_argument(abv, '--' + pros[i], type=types[i], help=help_file[pros[i]])
    parser.add_argument('-gi', '--GPU_INDEX', type=str,
                        help='Index of GPUs you want to use')
    args = parser.parse_args()
    log(args)
    
    # Instantiate config
    opt = Config()
    
    # Overwrite config with input args
    for k, v in vars(args).items():
        if v is not None and hasattr(opt, k):
            setattr(opt, k, v)
            log(k, v, getattr(opt, k))
    if args.GPU_INDEX:
        print(args.GPU_INDEX)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_INDEX
    main()
```



## Functions

### net.load()

Load the existing model.
:param model_type: temp model or best model.
:param map_location: your working environment.
    For loading the model file dumped from gpu or cpu.
:return: None.

### net.save()

Save the current model.
:param epoch:The current epoch (sum up). This will be together saved to file, aimed to keep tensorboard curve a continuous line when you train the net several times.
:param loss:Current loss.
:param name:The name of your saving file.
:return:None

### net.mt_save()

Save the model with a new thread. You can use this method in stead of self. save to save your model while not interrupting the training process, since saving big file is a time-consuming task. Also, this method will automatically record your best model and make a copy of it.
:param epoch: Current loss.
:param loss:
:return: None

### net=net.to_multi()

If you have multiple GPUs and you want to use them at the same time, you should call this method before training to send your model and data to multiple GPUs.
:return: None

**!!! Remember: Please use `net=net.to_multi()` to finish this command.**

### validate(net, val_loader)

Validate your model.
:param val_loader: A DataLoader class instance, which includes your validation data.
:return: validation loss and validation accuracy.

### predict(net, val_loader)

Make prediction based on your trained model. Please make sure you have trained your model or load the previous model from file.
:param val_loader: A DataLoader class instance, which includes your validation data.
:return: Prediction made.

### fit(net, train_loader, val_loader)

Training process. You can use this function to train your model. All configurations are defined and can be modified in config.py.
:param train_loader: A DataLoader class instance, which includes your train data.
:param val_loader: A DataLoader class instance, which includes your test data.
:return: None.



## Config

**USE_CUDA** : Whether to use GPU.

**LOAD_SAVED_MOD** : Whether to load saved model.

**SAVE_TEMP_MODEL** : Whether to save temporary model while training.

**SAVE_BEST_MODEL** : Whether to save best model while training.

**BEST_MODEL_BY_LOSS** : Evaluate whether a model is the optimal one by loss or accuracy.

**PRINT_BAD_CASE** : Whether to print the bad case while predicting.

**RUNNING_ON_JUPYTER** : Whether the program is running on a Jupyter Notebook.

**START_VOTE_PREDICT** : Whether to start vote predicting or training.

**START_PREDICT** : Whether to start predicting or training.

**TRAIN_ALL** : Whether to train in all of the data (train_set and val_set).

**TEST_ALL** : Whether to validate all of the data (train_set and val_set).

**TO_MULTI** : Whether to use multiple GPU, if available.

**ADD_SUMMARY** : Whether to add net graph into tensorboard summary.

**SAVE_PER_EPOCH** : Save your temp model every n epoch.

**BATCH_SIZE** : Batch size of training.

**VAL_BATCH_SIZE** : Batch size of validating.

**TENSOR_SHAPE** : Tensor shape of your input (batch dim is not included).

**DATALOADER_TYPE** : Dataloader type of your data (only `ImageFolder`, `SamplePairing`, `SixBatch`)

**OPTIMIZER** : Optimizer type. It is a string which is not case sensitive.Currently `Adam` and `SGD` are supported. Add new optimizer in the ./models/BasicModule.py -> get_optimizer()

**SGD_MOMENTUM** : The momentum if SDG is chosen as optimizer.

**TRAIN_DATA_RATIO** : The Train_Val data ratio.

**NUM_EPOCHS** : The epochs you want to train your model.

**NUM_CLASSES** : The number of your input data's class.

**NUM_VAL** : The number of your validation data.

**NUM_TRAIN** : The number of your train data.

**TOP_NUM** : If top n accuracy is ok for your result, put the `n` here.

**NUM_WORKERS** : Number of workers used in the DataLoader.

**CRITERION** : The Loss Class used in your training process, which is an instance of a Loss Class.

**LEARNING_RATE** : Learning rate used in your optimizer.

**TOP_VOTER** : Top `n` votes in the 6 picture generated will count for the final result.

**NET_SAVE_PATH** : Where to save your trained model.

**TRAIN_PATH** : Where your training set is located.

**VAL_PATH** : Where your validating set is located.

**CLASSES_PATH** : Where to save your classes' name.

**MODEL_NAME** : The name of your model.

**PROCESS_ID** : The `ID` of the current training process, which is the marker of the trained models. Please change it when some config or crucial code is altered!

**SUMMARY_PATH** : Where to save your tensorboard summary.

