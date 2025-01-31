from __future__ import division, print_function, annotations
import random
import time
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from typing import Any
from torch import inf

# import flavor


def get_unix_time(x):
    return int(time.mktime(x.timetuple()))



# =====================
#   Loggers and Meters
# =====================


# from the excellent https://github.com/pytorch/vision/blob/master/references/detection/utils.py
class Meter:
    """Track a series of values and provide access to a number of metric"""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque: Any = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

        self.M2 = 0
        self.mean = 0
        self.fmt = fmt

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.M2 = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def step(self, value):
        self.update(value)

    @property
    def var(self):
        return self.M2 / self.count if self.count > 2 else 0

    @property
    def sample_var(self):
        return self.M2 / (self.count - 1) if self.count > 2 else 0

    @property
    def median(self):
        return np.median(self.deque)

    @property
    def smoothed_avg(self):
        return np.mean(self.deque)

    @property
    def avg(self):
        return self.total / self.count

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.smoothed_avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter=" ", header="", print_freq=1, wandb=None, use_tqdm=False):
        self.meters = defaultdict(Meter)
        self.delimiter = delimiter
        self.print_freq = print_freq
        self.header = header
        self.wandb = wandb
        self.use_tqdm = use_tqdm

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"{k} is of type {type(v)}"
            self.meters[k].update(v)
        if self.wandb is not None:
            self.wandb.log(kwargs)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = [f"{name}: {str(meter)}" for name, meter in self.meters.items()]
        return self.delimiter.join(loss_str)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def step(self, iterable):
        start_time = time.time()
        end = time.time()
        iter_time = Meter(fmt="{avg:.4f}")
        data_time = Meter(fmt="{avg:.4f}")
        space_fmt = f":{len(str(len(iterable)))}d"
        pbar = tqdm(total=len(iterable))  # Initialize tqdm progress bar
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            eta_seconds = iter_time.global_avg * (len(iterable) - i)
            eta_string = str(timedelta(seconds=int(eta_seconds)))
            log_msg = self.delimiter.join(
                [
                    self.header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
            log_data = log_msg.format(
                i,
                len(iterable),
                eta=eta_string,
                meters=str(self),
                time=str(iter_time),
                data=str(data_time),
            )
            if self.use_tqdm:
                pbar.set_description(log_data)  # Update tqdm description
                pbar.update(1)  # Update tqdm progress
            else:
                print(log_data)  # Print log data if not using tqdm
            end = time.time()
        if self.use_tqdm:
            pbar.close()  # Close tqdm progress bar
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print(f"{self.header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


class ConvergenceMeter:
    """This is a modification of pytorch's ReduceLROnPlateau object
        (https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau)
        which acts as a convergence meter. Everything
        is the same as ReduceLROnPlateau, except it doesn't
        require an optimizer and doesn't modify the learning rate.
        When meter.converged(loss) is called it returns a boolean that
        says if the loss has converged.

    Args:
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity metered has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity metered has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> meter = Meter('min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     if meter.converged(val_loss):
        >>>         break
    """

    def __init__(
        self,
        mode="min",
        patience=10,
        verbose=False,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        eps=1e-8,
    ):
        self.has_converged = False
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def update(self, metrics, epoch=None):
        return self.step(metrics, epoch=None)

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.has_converged = True

        return self.has_converged

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError(f"threshold mode {threshold_mode} is unknown!")

        self.mode_worse = inf if mode == "min" else -inf
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


class BestMeter:
    """This is like ConvergenceMeter except it stores the
        best result in a set of results. To be used in a
        grid search

    Args:
        mode (str): One of `min`, `max`. In `min` mode, best will
            be updated when the quantity metered is lower than the current best;
            in `max` mode best will be updated when the quantity metered is higher
            than the current best. Default: 'max'.

    """

    def __init__(self, name="value", mode="max", object_name="epoch", verbose=True):
        self.has_converged = False
        self.verbose = verbose
        self.mode = mode
        self.name = name
        self.obj_name = object_name
        self.best = None
        self.best_obj = None
        self.mode_worse = None  # the worse value for the chosen mode
        self._init_is_better(mode=mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse

    def step(self, metrics, **kwargs):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.is_better(current, self.best):
            self.best = current
            self.best_obj = kwargs
            if self.verbose:
                print("*********New best**********")
                print(f"{self.name}: ", current)
                if self.best_obj:
                    print(f"{self.best_obj}")
                print("***************************")
            return True

        return False

    def is_better(self, a, best):
        return a < best if self.mode == "min" else a > best

    def _init_is_better(self, mode):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")
        self.mode_worse = inf if mode == "min" else -inf
        self.mode = mode


def detect_cuda(args):
    if "cuda" not in args.__dict__:
        return args
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_number}")
        args.cuda = True
    else:
        args.device = torch.device("cpu")
        args.cuda = False
    return args


def hits_and_misses(y_hat, y_testing):
    tp = sum(y_hat + y_testing > 1)
    tn = sum(y_hat + y_testing == 0)
    fp = sum(y_hat - y_testing > 0)
    fn = sum(y_testing - y_hat > 0)
    return tp, tn, fp, fn


def classification_metrics(labels, preds):
    tp, tn, fp, fn = hits_and_misses(labels, preds)

    precision = tp / (tp + fp) if (tp + fp) != 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) != 0 else np.nan

    f1 = 2.0 * (precision * recall / (precision + recall)) if (precision + recall) != 0 else np.nan

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "prec": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "acc": accuracy_score(preds, labels),
    }


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# from https://stackoverflow.com/questions/50246304/using-python-decorators-to-retry-request
def retry(times, exceptions, delay=1):
    """
    Retry Decorator
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param Exceptions: Lists of exceptions that trigger a retry attempt
    :type Exceptions: Tuple of Exceptions

    Example:
    @retry(times=3, exceptions=(ValueError, TypeError))
    def foo1():
        print('Some code here ....')
        print('Oh no, we have exception')
        raise ValueError('Some error')

    foo1()
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print("Exception thrown when attempting to run %s, attempt " "%d of %d" % (func, attempt, times))
                    print(e)
                    attempt += 1
                    time.sleep(delay)
            return func(*args, **kwargs)

        return newfn

    return decorator


def default_init(args):
    seed_all(args.seed)
    args = detect_cuda(args)
    args.output_dir = str(Path(args.output_dir).absolute())
    return args


def join_path(root, path):
    return str(Path("/".join([root, path])).absolute()) if path else path


def add_home(home_dir, *args) -> Any:
    if len(args) == 1:
        return join_path(home_dir, args[0])
    return [join_path(home_dir, p) for p in args]


def parse_comma_list(s: str):
    return list(map(lambda r: r.strip(), s.split(",")))


def string_to_list(*args):
    def _string_to_list(val):
        if isinstance(val, str):
            return list(map(int, val.split()))
        elif isinstance(val, list):
            return val
        elif isinstance(val, int):
            return [val]
        else:
            raise ValueError

    if len(args) == 1:
        return _string_to_list(args[0])
    return [_string_to_list(p) for p in args]
