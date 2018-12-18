# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch


class Config(object):
    def __init__(self):
        # Action definition
        self.USE_CUDA            = torch.cuda.is_available()
        self.LOAD_SAVED_MOD      = True
        self.SAVE_TEMP_MODEL     = True
        self.SAVE_BEST_MODEL     = True
        self.RUNNING_ON_JUPYTER  = False
        self.MASS_TESTING        = False
        self.START_PREDICT       = False
        self.START_EVAL          = False
        self.TRAIN_ALL           = False
        self.TEST_ALL            = False
        self.TO_MULTI            = False
        self.ADD_SUMMARY         = False
        self.SAVE_PER_EPOCH      = 1

        # Tensor shape definition
        self.BATCH_SIZE          = 64
        self.EVAL_BATCH_SIZE     = 64
        self.TENSOR_SHAPE        = (3, 224, 224)
        self.LINER_HID_SIZE      = 128

        # Program information
        self.CRITERION           = torch.nn.CrossEntropyLoss()
        self.DATALOADER_TYPE     = "ImageFolder"
        self.OPTIMIZER           = "Adam"
        self.TRAIN_DATA_RATIO    = 0.7
        self.NUM_EPOCHS          = 500
        self.NUM_CLASSES         = 250
        self.NUM_VAL             = 1
        self.NUM_TRAIN           = 1
        self.TOP_NUM             = 1
        self.NUM_WORKERS         = 0

        # Hyper parameters
        self.LEARNING_RATE       = 0.0001

        # Name and path definition
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.TRAIN_PATH          = "../cards_for_train"
        self.VAL_PATH            = "../cards_250_7/cards_for_val"
        self.CLASSES_PATH        = "./source/classes.json"
        self.MODEL_NAME          = "MobileNetV2"
        self.PROCESS_ID          = "Test"
        if self.TRAIN_ALL:
            self.PROCESS_ID += '_TRAIN_ALL'
        self.SUMMARY_PATH        = "./source/summary/"+self.MODEL_NAME+'_'+self.PROCESS_ID
