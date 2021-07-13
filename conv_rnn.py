import torch
from torch import Tensor, nn
from typing import Tuple, Callable, List

from progressiya.base import Progressive, InjectLast, TensorCollector, InjectByName, RNNProgressive, RNNElementwise
from progressiya.unet import ProgressiveSequential, ZapomniKak, CopyKwToArgs, InputFilterName, SetArgs
from view import View


class ConvLSTMCell(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, image_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = kernel_size // 2

        self.Wxi = torch.nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size,
                                   padding=self.padding)
        self.Whi = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,
                                   padding=self.padding, bias=False)
        self.register_parameter(name='Wci', param=torch.nn.Parameter(torch.empty(1, self.hidden_channels, image_size[0], image_size[1])))
        torch.nn.init.zeros_(self.Wci)

        self.Wxf = torch.nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size,
                                   padding=self.padding)
        self.Whf = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,
                                   padding=self.padding, bias=False)
        self.register_parameter(name='Wcf', param=torch.nn.Parameter(torch.empty(1, self.hidden_channels, image_size[0], image_size[1])))
        torch.nn.init.zeros_(self.Wcf)

        self.Wxc = torch.nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size,
                                   padding=self.padding)
        self.Whc = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,
                                   padding=self.padding, bias=False)

        self.Wxo = torch.nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size,
                                   padding=self.padding)
        self.Who = torch.nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size,
                                   padding=self.padding, bias=False)
        self.register_parameter(name='Wco', param=torch.nn.Parameter(torch.empty(1, self.hidden_channels, image_size[0], image_size[1])))
        torch.nn.init.zeros_(self.Wco)

    def forward(self, x, inner_states):
        h, c = inner_states

        # Wci, Wcf, Wco = self.Wci.to(x.device), self.Wci.to(x.device), self.Wci.to(x.device)

        it = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.Wci * c)
        ft = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.Wcf * c)
        ct = ft * c + it * torch.tanh(self.Wxc(x) + self.Whc(h))
        ot = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.Wco * ct)
        ht = ot * torch.tanh(ct)

        return ht, ct

    def init_hidden(self, batch_size, image_size):
        device = self.Wxi.weight.device

        h = torch.zeros(batch_size, self.hidden_channels, image_size[0], image_size[1], device=device)
        c = torch.zeros(batch_size, self.hidden_channels, image_size[0], image_size[1], device=device)

        return h, c


class FirstListCollector(TensorCollector[List[Tensor]]):

    def __init__(self):
        self.data = []

    def result(self) -> List[Tensor]:
        out = self.data
        self.data = []
        return out[1:]

    def append(self, t: Tuple[Tensor]) -> None:
        self.data.append(t[0])


class ConvLSTM(torch.nn.Module):
    """Shi, Xingjian & Chen, Zhourong & Wang, Hao & Yeung, Dit-Yan & Wong, Wai Kin & WOO, Wang-chun. (2015).
       Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.
       https://arxiv.org/pdf/1506.04214.pdf
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, image_size):
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = [hidden_channels] * num_layers
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.image_size = image_size
        self.cell_input_channels = [self.input_channels] + [hidden_channels] * (num_layers - 1)

        self.cells = []
        self.progression = []
        for i in range(0, self.num_layers):
            self.cells.append(ConvLSTMCell(input_channels=self.cell_input_channels[i],
                                           hidden_channels=self.hidden_channels[i],
                                           kernel_size=self.kernel_size,
                                           image_size=[image_size[0] // 2**i, image_size[1] // 2**i]))
            self.progression += [
                SetArgs({f"state_{i}"}),
                InputFilterName({"x"}),
                RNNProgressive[List[Tensor]](self.cells[i], InjectByName("inner_states"), FirstListCollector),
                ZapomniKak("x"),
                SetArgs({"x"}),
                InputFilterName(set()),
                RNNElementwise(nn.Conv2d(self.hidden_channels[i], self.hidden_channels[i], 4, 2, 1)),
                ZapomniKak("x"),
            ]

        self.progression = ProgressiveSequential(
            *(self.progression + [
                SetArgs({"x"}),
                InputFilterName(set()),
                RNNElementwise(nn.Sequential(
                View(-1),
                nn.Linear(hidden_channels * 4 * 4, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, 2),
                nn.Sigmoid()
            ))])
        )

        self.cells = nn.ModuleList(self.cells)

    def forward(self, x: List[Tensor]):

        init_states = {}

        for i in range(0, self.num_layers):
             init_states[f"state_{i}"] = self.cells[i].init_hidden(x[0].shape[0],
                                                                   [self.image_size[0] // 2**i, self.image_size[1] // 2**i])

        return self.progression.forward(x=x, **init_states)

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cells[i].init_hidden(batch_size, image_size))

        return init_states


if __name__ == "__main__":

    lstm = ConvLSTM(1, 10, 3, 4, [64, 64])

    x = torch.randn(4, 15, 1, 64, 64)
    xx = [x[:, t] for t in range(15)]

    res1 = lstm.forward(xx)
    print(len(res1), res1[0].shape)
