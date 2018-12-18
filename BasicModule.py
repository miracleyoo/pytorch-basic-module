# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import codecs
import datetime
import os
import socket

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from .BasicModuleSupporter import *


class BasicModule(nn.Module):
    """
        Basic pytorch module class. A wrapped basic model class for pytorch models.
        You can Inherit it to make your model easier to use. It contains methods
        such as load, save, multi-thread save, parallel distribution, train, validate,
        predict and so on.
    """

    def __init__(self, opt=None, device=None):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__
        self.opt = opt
        self.best_loss = 1e8
        self.epoch_fin = 0
        self.threads = []
        self.server_name = socket.getfqdn(socket.gethostname())
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': [], 'epoch': 0}
        self.writer = SummaryWriter(opt.SUMMARY_PATH)

    def load(self, model_type: str = "temp_model.dat", map_location=None) -> None:
        """
            Load the existing model.
            :param model_type: temp model or best model.
            :param map_location: your working environment.
                For loading the model file dumped from gpu or cpu.
            :return: None.
        """
        log('Now using ' + self.opt.MODEL_NAME + '_' + self.opt.PROCESS_ID)
        log('Loading model ...')
        if not map_location:
            map_location = self.device.type
        net_save_prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL_NAME + '_' + self.opt.PROCESS_ID + '/'
        temp_model_name = net_save_prefix + model_type
        if not os.path.exists(net_save_prefix):
            os.mkdir(net_save_prefix)
        if os.path.exists(temp_model_name):
            checkpoint = torch.load(temp_model_name, map_location=map_location)
            self.epoch_fin = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.history = checkpoint['history']
            self.load_state_dict(checkpoint['state_dict'])
            log("Load existing model: %s" % temp_model_name)
        else:
            log("The model you want to load (%s) doesn't exist!" % temp_model_name)

    def save(self, epoch, loss, name=None):
        """
        Save the current model.
        :param epoch:The current epoch (sum up). This will be together saved to file,
            aimed to keep tensorboard curve a continuous line when you train the net
            several times.
        :param loss:Current loss.
        :param name:The name of your saving file.
        :return:None
        """
        if loss < self.best_loss:
            self.best_loss = loss
        if self.opt is None:
            prefix = "./source/trained_net/" + self.model_name + "/"
        else:
            prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL_NAME + '_' + \
                     self.opt.PROCESS_ID + '/'
            if not os.path.exists(prefix): os.mkdir(prefix)

        if name is None:
            name = "temp_model.dat"

        path = prefix + name
        try:
            state_dict = self.module.state_dict()
        except:
            state_dict = self.state_dict()
        torch.save({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'best_loss': self.best_loss,
            'history': self.history
        }, path)

    def mt_save(self, epoch, loss):
        """
        Save the model with a new thread. You can use this method in stead of self.save to
        save your model while not interrupting the training process, since saving big file
        is a time-consuming task.
        Also, this method will automatically record your best model and make a copy of it.
        :param epoch: Current loss.
        :param loss:
        :return: None
        """
        if self.opt.SAVE_BEST_MODEL and loss < self.best_loss:
            log("Your best model is renewed")
        if len(self.threads) > 0:
            self.threads[-1].join()
        self.threads.append(MyThread(self.opt, self, epoch, self.best_loss, loss))
        self.threads[-1].start()
        if self.opt.SAVE_BEST_MODEL and loss < self.best_loss:
            log("Your best model is renewed")
            self.best_loss = loss

    def get_optimizer(self):
        """
        Get your optimizer by parsing your opts.
        :return:Optimizer.
        """
        if self.opt.OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.LEARNING_RATE)
        else:
            raise KeyError("==> The optimizer defined in your config file is not supported!")
        return optimizer

    def to_multi(self):
        """
        If you have multiple GPUs and you want to use them at the same time, you should
        call this method before training to send your model and data to multiple GPUs.
        :return: None
        """
        if torch.cuda.is_available():
            log("Using", torch.cuda.device_count(), "GPUs.")
            if torch.cuda.device_count() > 1:
                pmodel = torch.nn.DataParallel(self)
                attrs_p = [meth for meth in dir(pmodel) if not meth.startswith('_')]
                attrs = [meth for meth in dir(self) if not meth.startswith('_') and meth not in attrs_p]
                for attr in attrs:
                    setattr(pmodel, attr, getattr(self, attr))
                log("Using data parallelism.")
        else:
            log("Using CPU now.")
        pmodel.to(self.device)
        return pmodel

    def plot_history(self, figsize=(20, 9)):
        import matplotlib.pyplot as plt
        import seaborn as sns
        f, axes = plt.subplots(1, 2, figsize=figsize)
        sns.lineplot(range(1, self.epoch_fin + 1), self.history['train_acc'], label='Train Accuracy', ax=axes[0])
        sns.lineplot(range(1, self.epoch_fin + 1), self.history['val_acc'], label='Val Accuracy', ax=axes[0])
        sns.lineplot(range(1, self.epoch_fin + 1), self.history['train_loss'], label='Train Loss', ax=axes[1])
        sns.lineplot(range(1, self.epoch_fin + 1), self.history['val_loss'], label='Val Loss', ax=axes[1])
        plt.tight_layout()
        if hasattr(self.opt, 'RUNNING_ON_JUPYTER') and self.opt.RUNNING_ON_JUPYTER:
            plt.show()
        else:
            f.savefig(os.path.join(self.opt.SUMMARY_PATH + "history_output.jpg"))

    def write_summary(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        sum_path = os.path.join(self.opt.SUMMARY_PATH, 'Model_Record_Form.md')
        with codecs.open('./config.py', 'r', encoding='utf-8') as f:
            raw_data = f.readlines()
            configs = "|Config Name|Value|\n|---|---|\n"
            for line in raw_data:
                if line.strip().startswith('self.'):
                    pairs = line.strip().lstrip('self.').split('=')
                    configs += '|' + pairs[0] + '|' + pairs[1] + '|\n'
        with codecs.open('./models/Template.txt', 'r', encoding='utf-8') as f:
            template = ''.join(f.readlines())

        try:
            content = template % (
                self.model_name,
                current_time,
                self.server_name,
                self.history['epoch'],
                max(self.history['val_acc']),
                sum(self.history['val_acc']) / len(self.history['val_acc']),
                sum(self.history['val_loss']) / len(self.history['val_loss']),
                sum(self.history['train_acc']) / len(self.history['train_acc']),
                sum(self.history['train_loss']) / len(self.history['train_loss']),
                os.path.basename(self.opt.TRAIN_PATH),
                os.path.basename(self.opt.EVAL_PATH),
                self.opt.NUM_CLASSES,
                self.opt.CRITERION.__class__.__name__,
                self.opt.OPTIMIZER,
                self.opt.LEARNING_RATE,
                configs,
                str(self)
            )
            with codecs.open(sum_path, 'w+', encoding='utf-8') as f:
                f.writelines(content)
        except:
            raise KeyError("Template doesn't exist or it conflicts with your format.")
