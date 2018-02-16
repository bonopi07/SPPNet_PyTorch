import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def get_spp_layer(input_filter, tensor_size, max_level=1):
    num_batch_size = tensor_size[0]
    w = tensor_size[-1]
    # get optimized (filter size, stride)
    for level in range(1, max_level+1):
        stride, filter_size = 0, 0
        if level == 1:
            stride, filter_size = w, w
        else:
            _s, _f = 0.0, 0
            while True:
                while _s != ((w - _f) / (level - 1)):
                    _f += 1
                if _s <= _f:
                    stride, filter_size = int(_s), _f
                    _s, _f = _s + 1, 0
                else:
                    break
        # print('level {l}: filter size: {f}, stride: {s}'.format(l=level, f=filter_size, s=stride))

        maxpool = nn.MaxPool1d(filter_size, stride=stride)
        x = maxpool(input_filter)
        if level == 1:
            output_tensor = x.view(num_batch_size, -1)
        else:
            output_tensor = torch.cat((output_tensor, x.view(num_batch_size, -1)), 1)

    return output_tensor


if __name__ == '__main__':
    input_tensor = Variable(torch.Tensor(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])).view(1, 1, 14))

    result = get_spp_layer(input_tensor, list(input_tensor.size()), 4)
    print(result)
    pass

