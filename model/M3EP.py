import torch
from collections import OrderedDict
from torch import nn


def batch_to_device(batch, device):
    for key in batch[0].keys():
        batch[0][key] = batch[0][key].float()
        batch[0][key] = batch[0][key].to(device)

    batch[1] = batch[1].to(device)
    return batch


class M3EP(nn.Module):
    """
    Multi-Modal Model for Energy label Prediction
    """
    def __init__(self, config):
        super(M3EP, self).__init__()
        self.config = config

        # Year of construction submodule
        submodule_name = 'year_of_construction'
        output_size = config['submodules'][submodule_name]['output_size']
        setattr(self, submodule_name, nn.Sequential(OrderedDict([
            ('y_o_c_linear1', nn.Linear(1, output_size)),
            ('y_o_c_relu', nn.ReLU()),
        ])))

        # Registration date submodule
        submodule_name = 'registration_date'
        output_size = config['submodules'][submodule_name]['output_size']
        setattr(self, submodule_name, nn.Sequential(OrderedDict([
            ('reg_date_linear1', nn.Linear(4, output_size)),
            ('reg_date_relu', nn.ReLU()),
        ])))

        # Recorded date submodule
        submodule_name = 'recorded_date'
        output_size = config['submodules'][submodule_name]['output_size']
        setattr(self, submodule_name,
                nn.Sequential(OrderedDict([
                    ('rec_date_linear1', nn.Linear(4, output_size)),
                    ('rec_date_relu', nn.ReLU())])))

        # House number submodule
        submodule_name = 'house_number'
        output_size = config['submodules'][submodule_name]['output_size']
        setattr(self, submodule_name,
                nn.Sequential(OrderedDict([
                    ('house_number_linear1', nn.Linear(2, output_size)),
                    ('house_number_relu', nn.ReLU())])))

        # # House number addition submodule
        # submodule_name = 'house_number_addition'
        # output_size = config['hp']['submodules'][submodule_name + '_size']
        # setattr(self, submodule_name,
        #         nn.Sequential(OrderedDict([
        #             ('house_number_addition_linear1', nn.Linear(36, output_size)),
        #             ('house_number_addition_relu', nn.ReLU()),
        #             ('house_number_addition_global_average', nn.AvgPool2d(output_size))
        #         ])))

        # Purposes submodule
        submodule_name = 'purposes'
        output_size = config['submodules'][submodule_name]['output_size']
        setattr(self, submodule_name,
                nn.Sequential(OrderedDict([
                    ('purposes_linear1', nn.Linear(11, output_size)),
                    ('purposes_relu', nn.ReLU())
                ])))

        # Postal code submodule
        submodule_name = 'postal_code'
        output_size = config['submodules'][submodule_name]['output_size']
        setattr(self, submodule_name,
                nn.Sequential(OrderedDict([
                    ('postal_code_linear1', nn.Linear(6 * 36, output_size)),
                    ('postal_code_relu', nn.ReLU())
                ])))

        # Geometry submodule
        submodule_name = 'geometry'
        output_size = self.config['submodules'][submodule_name]['output_size']
        convnet_kernel_size = self.config['submodules'][submodule_name]['cnn_kernel_size']
        maxpool_kernel_size = self.config['submodules'][submodule_name]['maxpool_kernel_size']
        setattr(self, submodule_name,
                nn.Sequential(OrderedDict([
                    ('geometry_conv1d_1', nn.Conv1d(
                        in_channels=5,
                        out_channels=output_size,
                        kernel_size=convnet_kernel_size,
                        padding=convnet_kernel_size - 1
                    )),
                    ('geometry', nn.ReLU(inplace=True)),
                    # ('geometry_conv1d_2', nn.Conv1d(
                    #     in_channels=output_size,
                    #     out_channels=output_size,
                    #     kernel_size=convnet_kernel_size,
                    #     padding=convnet_kernel_size - 1
                    # )),
                    # ('geometry', nn.ReLU(inplace=True)),
                    ('geometry_maxpool', nn.MaxPool1d(kernel_size=maxpool_kernel_size)),
                    ('geometry_conv1d_3', nn.Conv1d(
                        in_channels=output_size,
                        out_channels=output_size,
                        kernel_size=convnet_kernel_size,
                        padding=convnet_kernel_size - 1
                    )),
                    ('geometry_avg_pooling', nn.AdaptiveAvgPool1d(1)),
                ])))

        # Final linear late fusion submodule
        submodule_size_sum = sum([config['submodules'][m]['output_size'] for m in config['submodules']])
        late_fusion_hidden_size = config['late_fusion']['hidden_size']
        self.late_fusion = nn.Sequential(OrderedDict([
            ('concatenated_late_fusion_linear', nn.Linear(submodule_size_sum, late_fusion_hidden_size)),
            ('late_fusion_relu', nn.ReLU()),
            ('late_fusion_output', nn.Linear(late_fusion_hidden_size, config['output_size']))
        ]))

    def get_submodule_output(self, batch, module_name):
        module = self._modules[module_name]
        module_vec = batch[0][module_name + '_vec']

        if len(module_vec.shape) == 1:  # In case of scalar values, such as the year of construction
            module_vec = module_vec.unsqueeze(dim=1)

        if module_name == 'geometry':
            module_vec = module_vec.permute(0, 2, 1)

        module_output = module(module_vec)

        if len(module_output.shape) == 3:
            module_output = module_output.squeeze(dim=2)

        # module_output.retain_grad()
        return module_output

    def forward(self, batch):
        # Create outputs for all the submodules and corresponding modal vectors
        submodule_outputs = tuple(self.get_submodule_output(batch, module_name)
                                  for module_name in self.config['submodules'])

        for sample in submodule_outputs:
            if not len(sample.shape) == 2:
                raise ValueError('Wrong tensor dimension', sample)

        concatenated = torch.cat(submodule_outputs, dim=1)
        output = self.late_fusion(concatenated)
        return output
