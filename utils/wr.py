
from typing import List, Tuple, Type, Optional, Callable, Any
from torch import nn, Tensor


from utils.loss_utils import Loss



class ItersCounter:

    def __init__(self):
        self.__iter = 0
        self.active = {}

    def update(self, iter):
        self.__iter = iter
        for k in self.active.keys():
            self.active[k] = True

    def get_iter(self, key: str):
        self.active[key] = False
        return self.__iter


class WR:

    counter = ItersCounter()
    writer = None
    l1_loss = nn.L1Loss()

    @staticmethod
    def L1(name: Optional[str]) -> Callable[[Tensor, Tensor], Loss]:

        if name:
            WR.counter.active[name] = True

        def compute(t1: Tensor, t2: Tensor):
            loss = WR.l1_loss(t1, t2)
            if name:
                if WR.counter.get_iter(name) % 10 == 0:
                    WR.writer.add_scalar(name, loss, WR.counter.get_iter(name))
            return Loss(loss)

        return compute

    @staticmethod
    def writable(name: str, f: Callable[[Any], Loss]):
        WR.counter.active[name] = True

        def decorated(*args, **kwargs) -> Loss:
            loss = f(*args, **kwargs)
            iter = WR.counter.get_iter(name)
            if iter % 10 == 0:
                WR.writer.add_scalar(name, loss.item(), iter)
            return loss

        return decorated