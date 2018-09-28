import re

import collections
import torch
from torch._six import int_classes, string_classes
from torch.utils.data.dataloader import numpy_type_map


def _pad_tensor(vec, pad, dim):
    """
    args:
        vec - pytorch or numpy tensor to pad
        pad - the integer size to pad to
        dim - integer dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    if pad_size[dim] == 0:
        return vec

    padding = torch.zeros(*pad_size)
    return torch.cat((torch.Tensor(vec), padding), dim=dim)


def dict_pad_collate(batch):
    """
    Basically a modified version of the default_collate function
    Puts each data field into a padded tensor with outer dimension batch size

    Use as:
    ```python
    train_loader = DataLoader(train_data,
                          batch_size=32,  # or something else to your liking
                          num_workers=8,  # or something else to your liking
                          collate_fn=dict_pad_collate)
    ```

    :param batch: a batch passed from an torch.utils.data.DataLoader instance
    :return: padded and collated tensor(s)
    """

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            max_len = max([x.shape[0] for x in batch])
            batch = [_pad_tensor(x, max_len, 0) for x in batch]
            return torch.stack(tuple(torch.Tensor(b) for b in batch), 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: dict_pad_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [dict_pad_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

