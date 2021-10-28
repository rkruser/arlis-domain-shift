"""
Metrics loggers
"""


import os
import datetime
import time
import pickle
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import utils

"""
For tracking scalar or possibly vector-valued metrics
"""
class Meter:
    def __init__(self, name, identity_object=0.0):
        self.name = name
        self.epochs = {}
        self.epoch_keys = []
        self.identity_object = identity_object
        self.overall_total = identity_object
        self.overall_count = 0

    def update(self, epoch, iteration, value, count, pre_averaged=False):
        if epoch not in self.epochs:
            self.epochs[epoch] = [[], [], [], 0, self.identity_object]
            self.epoch_keys.append(epoch)
        
        if pre_averaged:
            value *= count

        self.overall_count += count
        self.overall_total += value

        container = self.epochs[epoch]
        container[0].append(iteration)
        container[1].append(count)
        container[2].append(value)
        container[3] += count
        container[4] += value


    def average(self, epoch=-1, average_all=False):
        if average_all:
            if self.overall_count == 0:
                return self.identity_object
            return self.overall_total / self.overall_count
        else:
            container = self.epochs[self.epoch_keys[epoch]]
            if container[3] == 0:
                return self.identity_object
            return container[4] / container[3]
        
    def last_value(self):
        if len(self.epoch_keys) == 0:
            return 0.0
        container = self.epochs[self.epoch_keys[-1]]
        return container[2][-1] / container[1][-1]


    def __str__(self):
        return  repr(self) + '\n' + str(self.epochs)


    def __repr__(self):
        return self.name + "(total {0}, count {1}, epochs {2})".format(self.overall_total, self.overall_count, len(self.epochs))
        
"""
For tracking metrics extracted once an epoch, such FID score or reference images
"""
class Snapshot:
    def __init__(self, name):
        self.name = name
        self.epoch_keys = []
        self.items = []

    def add_snapshot(self, epoch, item):
        self.epoch_keys.append(epoch)
        self.items.append(item)

    def __repr__(self):
        return self.name + "(items {0})".format(len(self.items))

    def __str__(self):
        return repr(self) + '\n' + str(self.epoch_keys) + '\n' + str(self.items)



"""
Log all performance metrics
- Provides an interface for recording training progress
- Save printed output in a logfile
- Provides a function to save the current state of all metrics / extracted results
- Optionally send everything to tensorboard
"""
class Tracker:
    def __init__(self, load_state_dict=None, savefile='metrics.pth', savedir='models/', metadata=None,  use_tensorboard=False, use_logfile=False, tensorboard_logdir='./tensorboard_log', logdir='./', logfile='runlog.out'):
        if load_state_dict is not None:
            self.load_state_dict(load_state_dict)
        else:
            self.metadata = metadata
            self.meters = {}
            self.snapshots = {}
            self.message_log = []
            self.current_epoch = 0
            self.epoch_times = []
            self.current_iteration = 0
            self.current_mode = 'train'

        self.savefile = savefile
        self.savedir = savedir

        self.use_tensorboard = use_tensorboard 
        self.tensorboard_logdir = tensorboard_logdir
        
        if use_tensorboard:
#            if run_name is None:
#                run_name = str(datetime.datetime.now())

            self.tensorboard_writer = SummaryWriter(log_dir = tensorboard_logdir)


        self.use_logfile = use_logfile
        self.logdir = logdir
        self.logfile_name = logfile
        if use_logfile:
            self.logfile = open(os.path.join(logdir, logfile), "a")
            self.logfile.write("======== Run {0} =======\n".format(str(datetime.datetime.now())))
            self.logfile.write(str(metadata) + '\n')

        # Need to to this last after logfile is defined
        if load_state_dict is not None:
            self.log_message("Loaded tracker from dict")

    def register_meters(self, *args):
        for meter_name in args:
            if meter_name not in self.meters:
                self.meters[meter_name] = Meter(meter_name)

    def set_mode(self, mode, print_mode=True):
        self.current_mode = mode #e.g. "train" or "test"
        self.log_message(mode, print_message=print_mode)

    def print_current(self, *args, header=None, iteration_header=True, average_last_epoch = False):
        if header is not None:
            print(header) 

            if self.use_logfile:
                self.logfile.write(header+'\n')

        if iteration_header:
            printstr = "=={0} iter {1}==".format(self.current_mode, self.current_iteration)
            print(printstr)

            if self.use_logfile:
                self.logfile.write(printstr + '\n')

        meter_keys = args if len(args)>0 else self.meters.keys()
        for meter_name in meter_keys:
            if average_last_epoch:
                value = self.meters[meter_name].average(epoch=self.current_epoch)
            else:
                value = self.meters[meter_name].last_value()

            printstr = "  {0} = {1}".format(meter_name, value)
            print(printstr)

            if self.use_logfile:
                self.logfile.write(printstr + '\n')


    def update_epoch(self, epoch=None, print_averages=True, time=None, print_time=True):
        if print_averages:
            self.print_current(header="==== Epoch {0} ====".format(self.current_epoch), iteration_header=False, average_last_epoch=True)
        if time is not None:
            self.epoch_times.append((self.current_epoch, time))
            msg_str = "Epoch time " + str(time)
            self.log_message(msg_str, print_message=print_time)
            
        if self.use_tensorboard:
            for meter_name in self.meters:
                meter_epoch_average = self.meters[meter_name].average(epoch=self.current_epoch)
                self.tensorboard_writer.add_scalar(meter_name, meter_epoch_average, self.current_epoch)
        if epoch is None:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch

        # Later add in meter groups for plotting together


    def update_meters_from_easydict(self, edict, count, iteration, pre_averaged=False, tensorboard_log_iteration=False):
        walk = edict.walk(return_dict=True)
        for key in walk:
            self.update_meter(key, walk[key], count, iteration, pre_averaged=pre_averaged, tensorboard_log_iteration=tensorboard_log_iteration)

    def update_meter(self, meter_name, value, count, iteration, pre_averaged=False, tensorboard_log_iteration=False):
        self.current_iteration = iteration
        # Update current epoch and log epoch average to tensorboard if necessary
        if meter_name not in self.meters:
            self.meters[meter_name] = Meter(meter_name)

        self.meters[meter_name].update(self.current_epoch, iteration, value, count, pre_averaged=pre_averaged)

        # Update tensorboard for current batch
        if self.use_tensorboard and tensorboard_log_iteration:
            self.tensorboard_writer.add_scalar(meter_name+'_batches', value, iteration)

# Implement as needed
#    def snapshot_from_easydict(self, edict, epoch):
#        pass
        
        
    def snapshot(self, snapshot_name, epoch, item, tensorboard_type='image'):
        if snapshot_name not in self.snapshots:
            self.snapshots[snapshot_name] = Snapshot(snapshot_name)

        self.snapshots[snapshot_name].add_snapshot(epoch, item)

        if self.use_tensorboard and (tensorboard_type is not None):
            if tensorboard_type == 'image':
                self.tensorboard_writer.add_image(snapshot_name, item, epoch)
            elif tensorboard_type == 'histogram':
                self.tensorboard_writer.add_histogram(snapshot_name, item, epoch)


    def log_message(self, message, print_message=True):
        self.message_log.append( (message, self.current_epoch, self.current_iteration) ) #later add in python logging functionality
        if print_message:
            print(message)
        if self.use_logfile:
            self.logfile.write( message + '\n' )

    def state_dict(self):
        state_keys = ['metadata', 'meters', 'snapshots', 'message_log', 'current_epoch', 'current_iteration', 'savefile', 'savedir', 'use_tensorboard', 'tensorboard_logdir', 'use_logfile', 'logdir', 'logfile_name', 'current_mode', 'epoch_times']
        state_dict = { key: getattr(self, key) for key in state_keys }
        return utils.StateDictWrapper(state_dict)

    def save(self):
        with open(os.path.join(self.savedir, self.savefile), "wb") as savefile:
            pickle.dump(self.state_dict(),savefile)         

    def load_state_dict(self, state_dict):
        for key in state_dict: #because of the wrapper
            setattr(self, key, state_dict[key])
        

    


def test_tracker():
    tracker = Tracker(savefile='metrics_test.pth', savedir = './', metadata={'test':True}, use_logfile=True, 
                      logdir='./', logfile='runlog_test.out', use_tensorboard=True, tensorboard_logdir='tensorboard_logs')

    for i in range(30):
        tracker.set_mode('training')
        time1 = time.time()
        for j in range(100):
            tracker.update_meter('value/t1', np.sin(i*100+j), 1, i*100+j, tensorboard_log_iteration=True)
            tracker.update_meter('value/t2', 3*torch.randn(1).item(), 3, j)    
            if j % 50 == 0:
                tracker.print_current('value/t1', 'value/t2')

        tracker.set_mode('testing')
        for k in range(50):
            tracker.update_meter('testvalue/t1', -torch.rand(1).item(), 1, k)
            tracker.update_meter('testvalue/t2', torch.rand(5).mean().item(), 5, k, pre_averaged=True)
            if k%25 == 0:
                tracker.print_current('testvalue/t1', 'testvalue/t2')

        tracker.snapshot('general_snapshot', i, [i+3,i+4,i+5], tensorboard_type=None)
        tracker.snapshot('image_snapshot', i, torch.rand(3,64,64), tensorboard_type='image')
        tracker.update_meter('sinewave', np.sin(i), 1, i)
        tracker.save()
        time2 = time.time()

        tracker.update_epoch(print_averages=True, time=time2-time1)


def load_tracker():
    tracker_dict = pickle.load(open('metrics_test.pth', 'rb'))
    tracker = Tracker(load_state_dict = tracker_dict)
    print(tracker.meters, tracker.snapshots, tracker.epoch_times) #tracker.message_log, tracker.current_epoch, tracker.current_iteration)
#    print(str(tracker.meters['value/t1']))
#    print(str(tracker.snapshots['general_snapshot']))


if __name__ == '__main__':
    test_tracker()
    load_tracker()

    






