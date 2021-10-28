import utils
import performance_tracking

from datasets import collate_functions


import time

import torch






def run(model, dataloader, run_opts, tracker=None):
    if tracker is None:
        tracker = performance_tracking.Tracker(**run_opts.tracking_opts)
        tracker.set_mode("eval")

#    dataloader = torch.utils.data.Dataloader(dataset, batch_size=run_opts.batch_size, shuffle=run_opts.shuffle, drop_last=run_opts.drop_last, collate_fn = run_opts.collate_fn, pin_memory=run_opts.pin_memory)

    if run_opts.collect_outputs:
        all_outputs = []
    for iternum, batch in enumerate(dataloader):
        outputs, metrics = model.functions.run(model, batch)
        tracker.update_meters_from_easydict(metrics, batch.data_size(), pre_averaged=model.info.loss_is_averaged, tensorboard_log_iteration=run_opts.use_tensorboard)

        if run_opts.collect_outputs:
            all_outputs.append(outputs)

        if (iternum+1) % run_opts.printevery == 0:
            keys, _ = metrics.walk()
            tracker.print_current(*keys, iteration_header=True, average_last_epoch=run_opts.always_average_printouts)

    if run_opts.collect_outputs:
        pass # Something about collating
    else:
        all_outputs = None

    return all_outputs, tracker




def train(model, train_loader, train_opts, val_loader=None, tracker=None, run_opts=None):
    if tracker is None:
        tracker = performance_tracking.Tracker(metadata=model.info.opts, **train_opts.tracking_opts)
        tracker.register_meters(*train_opts.register_meters)
        tracker.set_mode('train')
        if train_opts.tracking_opts.load_state_dict is not None:
            epoch = tracker.current_epoch # go back and make more elegant later
            iternum = tracker.current_iteration + 1
        else:
            epoch = 0
            iternum = 0
    else:
        epoch = tracker.current_epoch
        iternum = tracker.current_iteration + 1

    model.functions.set_mode(model, 'train')
    dataiter = iter(train_loader)

#    time1 = time.time()
    timer = utils.Clock(total_ticks = train_opts.n_epochs)
    timer.start()
    while iternum < train_opts.n_iters and not model.functions.stop(model):
        batch = next(dataiter, None)
        if batch is None:
            if val_loader is not None:
                model.functions.set_mode('validation')
                tracker.set_mode('validation')
                run(model, val_loader, run_opts, tracker=tracker) #how does this interact with current_iternum?
                model.functions.set_mode('train')
                tracker.set_mode('train')
            if model.functions.snapshot is not None:
                snapshot = model.functions.snapshot(model)
                for key in snapshot:
                    shot = snapshot[key]
                    tracker.snapshot(key, epoch, shot.item, tensorboard_type=shot.dtype)

#            time2 = time.time()
            timer.tick()
            epoch += 1
            tracker.update_epoch(epoch=epoch, print_averages=True, time=timer.last_tick_time(), print_time=True)
#            time1 = time2
            remaining_seconds = timer.remaining_seconds()
            print("ETA: {0:8.2f} minutes / {1:8.2f} hours".format(remaining_seconds/60, remaining_seconds/3600))

            if (epoch+1) % train_opts.checkpoint_every == 0:
                print("Checkpointing")
                model.functions.save(model, epoch, iternum)
                tracker.save()

            if epoch == train_opts.n_epochs:
                break

            dataiter = iter(train_loader)
            batch = next(dataiter,None)

        metrics = model.functions.update(model, batch, epoch, iternum)
        tracker.update_meters_from_easydict(metrics, batch.data_size(), iternum, pre_averaged=model.info.loss_is_averaged, tensorboard_log_iteration=train_opts.use_tensorboard_per_iteration)

        if iternum % train_opts.print_every == 0:
            tracker.print_current(*train_opts.print_keys, iteration_header=True, average_last_epoch=train_opts.always_average_printouts)

        iternum += 1



    model.functions.save(model, epoch, iternum)
    tracker.save()




def simple_regressor_run(model, dataloader, run_opts):
    model.functions.set_mode(model,'eval')
    outputs = []
    for batch in dataloader:
        output, _ = model.functions.run(model,batch)
        outputs.append(output)
    
    return collate_functions.tensor_concat(outputs)

def invert_generator(model, dataloader, run_opts):
    model.functions.set_mode(model, 'eval')
    inverted = []

    timer = utils.Clock(total_ticks=len(dataloader))
    timer.start()
    for i,batch in enumerate(dataloader):
        inverted_batch =model.functions.invert(model, batch)
        inverted.append(inverted_batch)

        timer.tick()
        remaining_seconds = timer.remaining_seconds()
        print("Inverted {0} of {1}, batch time = {2:8.2f}, eta = {3:8.2f} seconds".format(i,len(dataloader), timer.last_tick_time(), remaining_seconds), end='\r')
        
    inverted = collate_functions.tensor_concat(inverted)

    return inverted



def calculate_jacobians(model, dataloader, run_opts):
    model.functions.set_mode(model, 'eval')
    jacobians = []
    time1 = time.time()
    moving_average = None

    timer = utils.Clock(total_ticks=len(dataloader))
    timer.start()
    for i,batch in enumerate(dataloader):
        jacobians_batch = model.functions.jacobian(model, batch)
        jacobians.append(jacobians_batch)


        timer.tick()
        remaining_seconds = timer.remaining_seconds()
        print("Processed Jacobians {0} of {1}, batch time = {2:8.2f}, eta = {3:6.2f} minutes / {4:4.2f} hours".format(i,len(dataloader), timer.last_tick_time(), remaining_seconds/60, remaining_seconds/3600), end='\r')

    jacobians = collate_functions.tensor_concat(jacobians)

    return jacobians








