# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import pickle
import shutil
import threading
import time

import numpy as np
from tqdm import tqdm


def validate(net, val_loader):
    """
    Validate your model.
    :param val_loader: A DataLoader class instance, which includes your validation data.
    :return: val loss and val accuracy.
    """
    net.eval()
    val_loss = 0
    val_acc = 0
    for i, data in tqdm(enumerate(val_loader), desc="Validating", total=len(val_loader), leave=False, unit='b'):
        inputs, labels, *_ = data
        inputs, labels = inputs.to(net.device), labels.to(net.device)

        # Compute the outputs and judge correct
        outputs = net(inputs)
        loss = net.opt.CRITERION(outputs, labels)
        val_loss += loss.item()

        predicts = outputs.sort(descending=True)[1][:, :net.opt.TOP_NUM]
        for predict, label in zip(predicts.tolist(), labels.cpu().tolist()):
            if label in predict:
                val_acc += 1
    return val_loss / net.opt.NUM_VAL, val_acc / net.opt.NUM_VAL


def predict(net, val_loader):
    """
    Make prediction based on your trained model. Please make sure you have trained
    your model or load the previous model from file.
    :param test_loader: A DataLoader class instance, which includes your test data.
    :return: Prediction made.
    """
    recorder = []
    log("Start predicting...")
    net.eval()
    for i, data in tqdm(enumerate(val_loader), desc="Validating", total=len(val_loader), leave=False, unit='b'):
        inputs, *_ = data
        inputs = inputs.to(net.device)
        outputs = net(inputs)
        predicts = outputs.sort(descending=True)[1][:, :net.opt.TOP_NUM]
        recorder.extend(np.array(outputs.sort(descending=True)[1]))
        pickle.dump(np.concatenate(recorder, 0), open("./source/test_res.pkl", "wb+"))
    return predicts

def fit(net, train_loader, val_loader):
    """
    Training process. You can use this function to train your model. All configurations
    are defined and can be modified in config.py.
    :param train_loader: A DataLoader class instance, which includes your train data.
    :param val_loader: A DataLoader class instance, which includes your test data.
    :return: None.
    """
    log("Start training...")
    epoch = 0
    optimizer = net.get_optimizer()
    for epoch in range(net.opt.NUM_EPOCHS):
        train_loss = 0
        train_acc = 0

        # Start training
        net.train()
        log('Preparing Data ...')
        for i, data in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader), leave=False,
                            unit='b'):
            inputs, labels, *_ = data
            inputs, labels = inputs.to(net.device), labels.to(net.device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = net.opt.CRITERION(outputs, labels)
            predicts = outputs.sort(descending=True)[1][:, :net.opt.TOP_NUM]
            for predict, label in zip(predicts.tolist(), labels.cpu().tolist()):
                if label in predict:
                    train_acc += 1

            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / net.opt.NUM_TRAIN
        train_acc = train_acc / net.opt.NUM_TRAIN

        # Start testing
        val_loss, val_acc = net.validate(val_loader)

        # Add summary to tensorboard
        net.writer.add_scalar("Train/loss", train_loss, epoch + net.epoch_fin)
        net.writer.add_scalar("Train/acc", train_acc, epoch + net.epoch_fin)
        net.writer.add_scalar("Eval/loss", val_loss, epoch + net.epoch_fin)
        net.writer.add_scalar("Eval/acc", val_acc, epoch + net.epoch_fin)
        net.history['train_loss'].append(train_loss)
        net.history['train_acc'].append(train_acc)
        net.history['val_loss'].append(val_loss)
        net.history['val_acc'].append(val_acc)

        # Output results
        log('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Eval Loss: %.4f, Eval Acc:%.4f'
            % (net.epoch_fin + epoch + 1, net.epoch_fin + net.opt.NUM_EPOCHS,
               train_loss, train_acc, val_loss, val_acc))

        # Save the model
        if epoch % net.opt.SAVE_PER_EPOCH == 0:
            net.mt_save(net.epoch_fin + epoch + 1, val_loss / net.opt.NUM_VAL)

    net.epoch_fin = net.epoch_fin + epoch + 1
    # net.plot_history()
    net.write_summary()
    log('Training Finished.')


def log(*args, end=None):
    if end is None:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]))
    else:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]),
              end=end)


class MyThread(threading.Thread):
    """
        Multi-thread support class. Used for multi-thread model
        file saving.
    """

    def __init__(self, opt, net, epoch, bs_old, loss):
        threading.Thread.__init__(self)
        self.opt = opt
        self.net = net
        self.epoch = epoch
        self.bs_old = bs_old
        self.loss = loss

    def run(self):
        lock.acquire()
        try:
            if self.opt.SAVE_TEMP_MODEL:
                self.net.save(self.epoch, self.loss, "temp_model.dat")
            if self.opt.SAVE_BEST_MODEL and self.loss < self.bs_old:
                self.net.best_loss = self.loss
                net_save_prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL_NAME + '_' + self.opt.PROCESS_ID + '/'
                temp_model_name = net_save_prefix + "temp_model.dat"
                best_model_name = net_save_prefix + "best_model.dat"
                shutil.copy(temp_model_name, best_model_name)
        finally:
            lock.release()


lock = threading.Lock()
