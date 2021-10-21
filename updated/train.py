import utils
import performance_tracking


import time

import torch






def run(model, dataset, run_opts, tracker=None):
    if tracker is None:
        tracker = performance_tracking.Tracker(**run_opts.tracking_opts)
        tracker.set_mode("eval")

    dataloader = torch.utils.data.Dataloader(dataset, batch_size=run_opts.batch_size, shuffle=run_opts.shuffle, drop_last=run_opts.drop_last, collate_fn = dataset.collate_fn)

    if run_opts.collect_outputs:
        all_outputs = []
    for iternum, batch in enumerate(dataloader):
        outputs, metrics = model.functions.run(model, batch)
        tracker.update_meters_from_easydict(metrics, batch.data_size(), pre_averaged=model.info.loss_is_averaged, tensorboard_log_iteration=run_opts.use_tensorboard)

        if run_opts.collect_outputs:
            all_outputs.append(outputs)

        if iternum % run_opts.printevery == 0:
            keys, _ = metrics.walk()
            tracker.print_current(*keys, iteration_header=True, average_last_epoch=run_opts.always_average_printouts)

    if run_opts.collect_outputs:
        pass # Something about collating
    else:
        all_outputs = None

    return all_outputs, tracker


"""
model.state contains: #accessible only through model.functions, so can have any form
 - network class names
 - networks
    - Each network constructed with **kwargs
    - All networks take DataDicts/EasyDicts as input and produce Datadicts/EasyDicts as output
    - Actually, previous line not necessary if all models are accessed through model.functions
 - optimizers
 - schedulers
 - device(s) to run on / parallelism info (maybe have separate "compute" section)
 - metadata about hyperparameters and file names
 - other info about current state
model.functions contains:
 - set_mode
 - snapshot (optionally; otherwise set this to None)
 - sample (optionally, for use in snapshot or alone)
 - update
    - Should do all the hard work
    - Should update model state / step parameters and everything
 - run
    - Run the model on some data without updating
 - save
    - should save each network separately with its optimizers/hyperparams/class names
 - stop
 - Hidden functions for loss/metrics, stepping gradient, stepping learning rate, and so forth,
    for use inside update
 - Should probably have something for initializing / loading maybe (or do that outside the model?)
model.info contains:
 - loss_is_averaged


dataset objects satisfy:
- Always return DataDict batches mapping to tensors (or EasyDict batches with some kind of batch_size parameter)
- Have a collate_fn attribute so dataloader can collate the type of data returned
to_implement:
 - A nice way of saving/loading data sampled data and constructing datasets out of it
 - (A nice way of loader a saved tracker and visualizing all its info)


train_opts contains:
 - n_iters
 - n_epochs
 - batch_size
 - shuffle
 - drop_last
 - run_opts # a dict of opts for validation running, if there is a validation set
 - checkpoint_every
 - printevery
 - use_tensorboard_per_iteration
 - tracking_opts
    - stuff about tensorboard
    - stuff about logfiles
"""
def train(model, train_dataset, train_opts, val_dataset=None, tracker=None, run_opts=None):
    if tracker is None:
        tracker = performance_tracking.Tracker(metadata=model.info.opt, **train_opts.tracking_opts)
        tracker.set_mode('train')

    model.functions.set_mode('train')
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = train_opts.batch_size, shuffle=train_opts.shuffle, drop_last=train_opts.drop_last, collate_fn = train_dataset.collate_fn)
    dataiter = iter(dataloader)
    epoch = 0
    iternum = 0
    time1 = time.time()
    while iternum < train_opts.n_iters and not model.functions.stop(model):
        batch = next(dataiter, None)
        if batch is None:
            if val_dataset is not None:
                model.functions.set_mode('validation')
                tracker.set_mode('validation')
                run(model, val_dataset, run_opts, tracker=tracker)
                model.functions.set_mode('train')
                tracker.set_mode('train')
            if model.functions.snapshot is not None:
                snapshot = model.functions.snapshot(model)
                for key in snapshot:
                    shot = snapshot[key]
                    tracker.snapshot(key, epoch, shot.item, tensorboard_type=shot.dtype)

            time2 = time.time()
            epoch += 1
            tracker.update_epoch(epoch=epoch, print_averages=True, time=time2-time1, print_time=True)
            time1 = time2

            if (epoch+1) % train_opts.checkpoint_every == 0:
                print("Checkpointing")
                model.functions.save(model, epoch, iternum)
                tracker.save()

            if epoch == train_opts.n_epochs:
                break

            dataiter = iter(dataloader)
            batch = dataiter.next()

        metrics = model.functions.update(model, batch, epoch, iternum)
        tracker.update_meters_from_easydict(metrics, batch.data_size(), pre_averaged=model.info.loss_is_averaged, tensorboard_log_iteration=train_opts.use_tensorboard_per_iteration)

        if iternum % train_opts.printevery == 0:
            keys, _ = metrics.walk()
            tracker.print_current(*keys, iteration_header=True, average_last_epoch=train_opts.always_average_printouts)

        iternum += 1



    model.functions.save(model, epoch, iternum)
    tracker.save()








"""
Todo:
Make collating work with DataDict
Add a ModelDict class
Write exact specifications for train/run function so you know what you need to implement for model and constraints thereon
Write model functions sufficient to allow training/sampling/snapshotting <--- you are here
Populate the default parameters for everything
Write a model save/load constructor for a basic GAN
Test everything
"""























