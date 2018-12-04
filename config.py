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
        self.MASS_TESTING        = False
        self.TRAIN_ALL           = False
        self.USE_NEW_DATA        = False
        self.SAVE_EVERY          = 1

        # Tensor shape definition
        self.BATCH_SIZE          = 256
        self.TEST_BATCH_SIZE     = 256
        self.NUM_CHANNEL         = 2
        self.LENGTH              = 41
        self.WIDTH               = 9
        self.LINER_HID_SIZE      = 128

        # Program information
        self.CRITERION           = torch.nn.CrossEntropyLoss()
        self.OPTIMIZER           = "Adam"
        self.TRAIN_DATA_RATIO    = 0.7
        self.NUM_EPOCHS          = 100
        self.NUM_CLASSES         = 369
        self.NUM_TEST            = 1
        self.NUM_TRAIN           = 1
        self.TOP_NUM             = 1
        self.NUM_WORKERS         = 0

        # Hyper parameters
        self.LEARNING_RATE       = 0.001

        # Name and path definition
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.MODEL               = 'MiracleWeightWideNet'
        self.PROCESS_ID          = 'PADDING_LOSS1-2_WEI4-2-1-1-NEW_GEN'
        if self.TRAIN_ALL:
            self.PROCESS_ID += '_TRAIN_ALL'
        self.SUMMARY_PATH        = "./source/summary/"+self.MODEL+'_'+self.PROCESS_ID+'_'
