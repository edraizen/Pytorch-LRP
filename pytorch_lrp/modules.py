import os
import torch
import time
import numpy as np
import torch.nn.functional as F
import psutil

from .rules import RULES

ALLOWED_LAYERS = []
ALLOWED_LAYERS_BY_NAME = {}

ALLOWED_PASS_LAYERS = [c.__name__ for c in (
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
    torch.nn.ELU, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d,
    torch.nn.Softmax, torch.nn.Sigmoid)]

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.3f%s%s" % (num, 'Yi', suffix)

def cpu_usage():
    return sizeof_fmt(psutil.Process(os.getpid()).memory_info().rss)

def gpu_usage():
    if torch.cuda.is_available():
        return sizeof_fmt(torch.cuda.memory_allocated())
    else:
        return "(No GPU)"

class LRPLayer(object):
    #Layer specific, subclasses can change these
    RULE = None
    NORMALIZE_BEFORE = False #True
    NORMALIZE_AFTER = False #True

    LAYER_NUM = None
    N_LAYERS = None

    #Normalization parameters
    NORMALIZE_METHOD = "fraction_clamp_filter"
    FRACTION_PASS_FILTER = [[0.0,0.6], [-0.6,-0.0]] #Postive, negetive
    FRACTION_CLAMP_FILTER = [[0.0,0.6], [-0.6,-0.0]] #Postive, negetive

    LRP_EXPONENT = 1.
    EPS = 1e-6
    IGNORE_UNSUPPORTED_LAYERS = False
    EPSILON = 0.25
    GAMMA = 0.25
    BETA = 0.5

    @classmethod
    def __init_subclass__(cls, layer_class=None, **kwds):
        super().__init_subclass__(*kwds)
        global ALLOWED_LAYERS
        global ALLOWED_LAYERS_BY_NAME

        if layer_class is not None:
            layer_name = cls.__name__ if layer_class is None else layer_class.__name__
            ALLOWED_LAYERS_BY_NAME[layer_name] = cls

        #If its an LRPPassLayer, it might have allowed pass modules
        if hasattr(cls, "ALLOWED_PASS_LAYERS") and isinstance(
          cls.ALLOWED_PASS_LAYERS, (list, tuple)):
            global ALLOWED_PASS_LAYERS
            ALLOWED_PASS_LAYERS += [c.__name__ for c in cls.ALLOWED_PASS_LAYERS]
            del cls.ALLOWED_PASS_LAYERS

    @classmethod
    def forward_pass_(cls, m, in_tensor: torch.Tensor,
      out_tensor: torch.Tensor):
        """Run and time forward_hh=hook method"""
        start = time.time()
        out = cls.forward_hook(m, in_tensor, out_tensor)
        end = time.time()
        print("Module {} took {} seconds".format(m, end-start))
        return out

    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        return

    @staticmethod
    def clean(m, force=False):
        return

    @classmethod
    def relprop_(cls, m, relevance_in, clean=True):
        if cls.NORMALIZE_BEFORE:
            relevance_in = cls.relnormalize(relevance_in)
        name = str(m).split("\n")[0]
        if issubclass(cls, LRPFunctionLayer):
            if cls.RULE is None:
                relevance = RULES["LayerNumRule"].relprop(cls, m, relevance_in, LRPLayer.N_LAYERS)
            else:
                relevance = RULES.get(cls.RULE, "ERule").relprop(cls, m, relevance_in)
        elif cls.RULE is not None and cls.RULE in RULES:
            relevance = RULES[cls.RULE].relprop(cls, m, relevance_in)
        else:
            relevance = cls.relprop(m, relevance_in)

        if cls.NORMALIZE_AFTER:
            relevance = cls.relnormalize(relevance)

        del relevance_in
        #setattr(m, "relevance", relevance)

        cls.clean(m, clean)

        torch.cuda.empty_cache()

        return relevance

    @classmethod
    def relprop(cls, m, relevance_in):
        return relevance_in.detach()

    @classmethod
    def relnormalize(cls, relevance):
        """Modified from github repo etjoa003/medical_imaging:
        https://github.com/etjoa003/medical_imaging/blob/master/isles2017/models/networks.py"""
        if isinstance(relevance, (list, tuple)):
            out = [cls.relnormalize(r) for r in relevance]
            del relevance
            return out

        if cls.NORMALIZE_METHOD in [None, 'raw']:
            return relevance.detach()
        elif cls.NORMALIZE_METHOD == 'standard':
            # this is bad. the sum can go too large
            with torch.no_grad():
                ss = torch.sum(R**2)**0.5
                normalized_rel = relevance/ss
                del ss, relevance
        elif cls.NORMALIZE_METHOD == 'fraction_pass_filter':
            with torch.no_grad():
                ss = torch.max(torch.FloatTensor.abs(relevance))
                normalized_rel = relevance/ss
                del relevance
                ppf = cls.FRACTION_PASS_FILTER[0]
                npf = cls.FRACTION_PASS_FILTER[1]
                Rplus = normalized_rel*(normalized_rel>=ppf[0]).to(torch.float)*(normalized_rel<=ppf[1]).to(torch.float)
                Rmin = normalized_rel*(normalized_rel>=npf[0]).to(torch.float)*(normalized_rel<=npf[1]).to(torch.float)
                normalized_rel = ss*(Rmin+Rplus)
                del ss, Rplus, Rmin
        elif cls.NORMALIZE_METHOD == 'fraction_clamp_filter':
            with torch.no_grad():
                ss = torch.max(torch.FloatTensor.abs(relevance))
                normalized_rel = relevance/ss
                del relevance
                ppf = cls.FRACTION_CLAMP_FILTER[0]
                npf = cls.FRACTION_CLAMP_FILTER[1]
                Rplus = torch.clamp(normalized_rel,min=ppf[0], max=ppf[1])*(normalized_rel>=0).to(torch.float)
                Rmin = torch.clamp(normalized_rel,min=npf[0], max=npf[1])*(normalized_rel<0).to(torch.float)
                normalized_rel = ss*(Rmin+Rplus)
                del ss, Rplus, Rmin
        else:
            raise Exception('Invalid normalization method {}'.format(self.NORMALIZE_METHOD))

        return normalized_rel.detach()

    @staticmethod
    def get(layer):
        #print(globals())
        try:
            #Class is key
            return ALLOWED_LAYERS_BY_NAME[layer.__class__]
        except KeyError:
            try:
                #Full name of class in allowed layers => manually mapped
                return ALLOWED_LAYERS_BY_NAME[layer.__class__.__name__]
            except KeyError:
                try:
                    #Check if the last parts of name match
                    name = layer.__class__.__name__.rsplit(".", 1)[-1]
                    return ALLOWED_LAYERS_BY_NAME[name]
                except KeyError:
                    if layer.__class__.__name__ in ALLOWED_PASS_LAYERS:
                        return LRPPassLayer
                    if hasattr(layer, "children") and list(layer.children()):
                        #Starting module, run through all children
                        return Module
                    elif hasattr(layer, "_modules") and list(layer._modules):
                        #Unknown module but has sub modules
                        return Container
                    elif not LRPLayer.ignore_unsupported_layers:
                        return LRPPassLayer
                    else:
                        raise NotImplementedError("The network contains layers that"
                                                  " are currently not supported {0:s}".format(str(layer)))

class LRPPassLayer(LRPLayer):
    #Add layer classes here to ignore them
    ALLOWED_PASS_LAYERS = []

class LRPFunctionLayer(LRPLayer):
    """Placeholder to distiguish function or Sequential modules"""
    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        setattr(m, 'in_shape', in_tensor[0].size())
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', out_tensor.size())
        return None

    @staticmethod
    def clean(m, force=False):
        if force:
            del m.in_shape, m.in_tensor, m.out_shape

    @staticmethod
    def reshape(a, b):
        return a, b

class Module(LRPLayer, layer_class=torch.nn.Module):
    @classmethod
    def relprop(cls, module, relevance_in):
        with torch.no_grad():
            for m in reversed(list(module.children())):
                relevance_in = LRPLayer.get(m).relprop_(m, relevance_in).detach()
        return relevance_in.detach()

class Container(LRPLayer, layer_class=torch.nn.Container):
    @classmethod
    def relprop(cls, module, relevance_in):
        with torch.no_grad():
            for m in reversed(module._modules.values()):
                relevance_in = LRPLayer.get(m).relprop_(m, relevance_in)
            return relevance_in

class Sequential(Container, layer_class=torch.nn.Sequential):
    pass

class ModuleList(Container, layer_class=torch.nn.ModuleList):
    pass

class ModuleDict(Container, layer_class=torch.nn.ModuleDict):
    pass

class LogSoftmax(LRPLayer, layer_class=torch.nn.LogSoftmax):
    @classmethod
    def relprop(cls, m, relevance_in):
        # Only layer that does not conserve relevance. Mainly used
        # to make probability out of the log values. Should probably
        # be changed to pure passing and the user should make sure
        # the layer outputs are sensible (0 would be 100% class probability,
        # but no relevance could be passed on).
        with torch.no_grad():
            if relevance_in.sum() < 0:
                relevance_in[relevance_in == 0] = -1e6
                relevance_in = relevance_in.exp()
            return relevance_in.detach()

class ReLU(LRPLayer, layer_class=torch.nn.ReLU):
    @staticmethod
    def backward_hook(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        return (torch.clamp(grad_in[0], min=0.0),)

class Linear(LRPFunctionLayer, layer_class=torch.nn.Linear):
    @staticmethod
    def forward_pass(m, tensor_in, weight):
        return F.linear(tensor_in, weight, bias=None).detach()

    @staticmethod
    def backward_pass(m, tensor, weight):
        return F.linear(tensor, weight.t(), bias=None).detach()

class _MaxPoolNd(LRPFunctionLayer):
    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        # Save the return indices value to make sure
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "indices", indices)
        setattr(m, 'weights', None)

        return super().forward_hook(m, in_tensor, out_tensor)

class MaxPool1d(_MaxPoolNd, layer_class=torch.nn.MaxPool1d):
    @staticmethod
    def forward_pass(m, tensor, weight, bias=None):
        with torch.no_grad():
            return F.max_pool1d(tensor, m.kernel_size, m.stride,
                                m.padding, m.dilation, m.ceil_mode,
                                m.return_indices).detach()

    @staticmethod
    def backward_pass(m, tensor, weight):
        with torch.no_grad():
            inverted = F.max_unpool1d(
                tensor, m.indices, m.kernel_size, m.stride, m.padding,
                output_size=m.in_shape)
            del layer_instance.indices, layer_instance.weights
        return inverted.detach()

class MaxPool2d(_MaxPoolNd, layer_class=torch.nn.MaxPool1d):
    @staticmethod
    def forward_pass(m, tensor, weight, bias=None):
        with torch.no_grad():
            return F.max_pool2d(tensor, m.kernel_size, m.stride,
                                m.padding, m.dilation, m.ceil_mode,
                                m.return_indices)

    @staticmethod
    def backward_pass(m, tensor, weight):
        inverted = F.max_unpool2d(
            tensor, m.indices, m.kernel_size, m.stride, m.padding,
            output_size=m.in_shape)
        del layer_instance.indices
        return inverted.detach()

class MaxPool3d(_MaxPoolNd, layer_class=torch.nn.MaxPool1d):
    @staticmethod
    def forward_pass(m, tensor, weight, bias=None):
        return F.max_pool3d(tensor, m.kernel_size, m.stride,
                            m.padding, m.dilation, m.ceil_mode,
                            m.return_indices)

    @staticmethod
    def backward_pass(m, tensor, weight):
        inverted = F.max_unpool3d(
            tensor, m.indices, m.kernel_size, m.stride, m.padding,
            output_size=m.in_shape)
        del layer_instance.indices
        return inverted.detach()

class Conv1d(LRPFunctionLayer, layer_class=torch.nn.Conv1d):
    @staticmethod
    def forward_pass(layer, in_tensor, weight, scale_groups=1):
        return F.conv1d(in_tensor, weight, bias=bias, stride=layer.stride,
            padding=layer.padding, dilation=layer.dilation,
            groups=scale_groups*layer.groups).detach()

    @staticmethod
    def backward_pass(layer, in_tensor, weight):
        return F.conv_transpose1d(in_tensor, weight,
            stride=layer.stride, padding=layer.padding,
            output_padding=layer.output_padding, groups=layer.groups,
            dilation=layer.dilation).detach()

class Conv2d(LRPFunctionLayer, layer_class=torch.nn.Conv2d):
    @staticmethod
    def forward_pass(layer, in_tensor, weight, scale_groups=1):
        return F.conv2d(in_tensor, weight, bias=bias, stride=layer.stride,
            padding=layer.padding, dilation=layer.dilation,
            groups=scale_groups*layer.groups).detach()

    @staticmethod
    def backward_pass(layer, in_tensor, weight):
        return F.conv_transpose2d(in_tensor, weight,
            stride=layer.stride, padding=layer.padding,
            output_padding=layer.output_padding, groups=layer.groups,
            dilation=layer.dilation).detach()

class Conv3d(LRPFunctionLayer, layer_class=torch.nn.Conv3d):
    @staticmethod
    def forward_pass(layer, in_tensor, weight, scale_groups=1):
        return F.conv3d(in_tensor, weight, stride=layer.stride,
            padding=layer.padding, dilation=layer.dilation,
            groups=scale_groups*layer.groups).detach()

    @staticmethod
    def backward_pass(layer, in_tensor, weight):
        return F.conv_transpose3d(in_tensor, weight,
            stride=layer.stride, padding=layer.padding,
            output_padding=layer.output_padding, groups=layer.groups,
            dilation=layer.dilation).detach()

class TransposeConvolution1D(LRPFunctionLayer, layer_class=torch.nn.ConvTranspose1d):
    """https://github.com/etjoa003/medical_imaging/blob/master/isles2017/models/networks_LRP.py"""
    @staticmethod
    def forward_pass(m, in_tensor, wieght):
        return F.conv_transpose1d(in_tensor, weight, stride=m.stride,
            padding=m.padding, output_padding=m.output_padding,
            dilation=m.dilation, groups=m.groups).detach()

    @staticmethod
    def backward_pass(m, in_tensor, wieght):
        return F.conv1d(in_tensor, weight, stride=m.stride, padding=m.padding,
            dilation=m.dilation, groups=m.groups).detach()

class TransposeConvolution2D(LRPFunctionLayer, layer_class=torch.nn.ConvTranspose2d):
    """https://github.com/etjoa003/medical_imaging/blob/master/isles2017/models/networks_LRP.py"""
    @staticmethod
    def forward_pass(m, in_tensor, wieght):
        return F.conv_transpose2d(in_tensor, weight, stride=m.stride,
            padding=m.padding, output_padding=m.output_padding,
            dilation=m.dilation, groups=m.groups).detach()

    @staticmethod
    def backward_pass(m, in_tensor, wieght):
        return F.conv2d(in_tensor, weight, stride=m.stride, padding=m.padding,
            dilation=m.dilation, groups=m.groups).detach()

class TransposeConvolution3D(LRPFunctionLayer, layer_class=torch.nn.ConvTranspose3d):
    """https://github.com/etjoa003/medical_imaging/blob/master/isles2017/models/networks_LRP.py"""
    @staticmethod
    def forward_pass(m, in_tensor, wieght):
        return F.conv_transpose3d(in_tensor, weight, stride=m.stride,
            padding=m.padding, output_padding=m.output_padding,
            dilation=m.dilation, groups=m.groups).detach()

    def backward_pass(m, in_tensor, wieght):
        return F.conv3d(in_tensor, weight, stride=m.stride, padding=m.padding,
            dilation=m.dilation, groups=m.groups).detach()

def relprop_size_adjustment(Z,R):
    sZ, sR = Z.size(), R.size()
    if not np.all(sZ==sR):
        tempR, tempZ = get_zero_container(Z,R), get_zero_container(Z,R)
        tempR[:sR[0],:sR[1],:sR[2],:sR[3],:sR[4]] = R; # R = tempR
        tempZ[:sZ[0],:sZ[1],:sZ[2],:sZ[3],:sZ[4]] = Z; # Z = tempZ
    else:
        return Z, R
    return tempZ, tempR

def get_zero_container(x,y):
    """
    Assume x, y are tensors of the same dimension
    but not necessarily have the same size
    for example x can be 3,4,5 and y 4,3,5
    return the size that can contain both: 4,4,5

    Example:
    x = torch.tensor(np.random.normal(0,1,size=(3,4,5)))
    y = torch.tensor(np.random.normal(0,1,size=(4,3,5)))
    z = get_zero_container(x,y)
    print(z.shape) # torch.Size([4, 4, 5])
    """
    return torch.zeros([max(sx, sy) for sx,sy in zip(x.size(),y.size())])
