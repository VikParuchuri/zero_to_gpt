from torch import nn

def get_module_list(count, module, *args, **kwargs):
    return nn.ModuleList([module(*args, **kwargs) for _ in range(count)])