import torch
import time
import numpy as np
import torch.nn.functional as F

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))

class LRPLayer(object):
    CURRENT_METHOD = None
    LRP_EXPONENT = 1.
    EPS = 1e-6
    ignore_unsupported_layers = False
    AVAILABLE_METHODS = []
    ALLOWED_LAYERS = []
    ALLOWED_LAYERS_BY_NAME = {}
    AVAILABLE_METHODS = set()
    ALLOWED_PASS_LAYERS = [
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.ELU, Flatten,
        torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d,
        torch.nn.Softmax, torch.nn.LogSoftmax, torch.nn.Sigmoid]

    @classmethod
    def __init_subclass__(cls, layer_class=None, **kwds):
        super().__init_subclass__(*kwds)
        LRPLayer.ALLOWED_LAYERS.append(cls)
        if layer_class is None:
            layer_class = cls.__name__.rsplit(".", 1)[-1]
        LRPLayer.ALLOWED_LAYERS_BY_NAME[layer_class] = cls
        LRPLayer.AVAILABLE_METHODS |= set(cls.AVAILABLE_METHODS)

        #If its an LRPPassLayer, it might have allowed pass modules
        if hasattr(cls, "ALLOWED_PASS_LAYERS") and isinstance(
          cls.ALLOWED_PASS_LAYERS, (list, tuple)):
            LRPLayer.ALLOWED_PASS_LAYERS += cls.ALLOWED_PASS_LAYERS

    def forward_(self, m, in_tensor: torch.Tensor,
      out_tensor: torch.Tensor):
        start = time.time()
        self.forward(m, in_tensor, out_tensor)
        end = time.time()
        print("Module {} took {} seconds".format(m, end-start))

    def forward(self, m, in_tensor: torch.Tensor,
      out_tensor: torch.Tensor):
        return None

    def relprop_(self, m, relevance_in):
        relevance = self.relprop(m, relevance_in)
        setattr(m, "relevance", relevance)
        return relevance

    def relprop(self, m, relevance_in):
        return relevance_in.detach()

    @staticmethod
    def get(layer):
        try:
            #Class is key
            return LRPLayer.ALLOWED_LAYERS_BY_NAME[layer.__class__]()
        except KeyError:
            try:
                #Full name of class in allowed layers => manually mapped
                return LRPLayer.ALLOWED_LAYERS_BY_NAME[layer.__class__.__name__]()
            except KeyError:
                try:
                    #Check if the last parts of name match
                    name = layer.__class__.__name__.rsplit(".", 1)[-1]
                    return LRPLayer.ALLOWED_LAYERS_BY_NAME[name]()
                except KeyError:
                    if hasattr(layer, "children"):
                        #Starting module, run through all children
                        return Module()
                    elif hasattr(layer, "_modules"):
                        #Unknown module but has sub modules
                        return Container()
                    elif layer.__class__ in LRPLayer.ALLOWED_PASS_LAYERS or \
                      not LRPLayer.ignore_unsupported_layers:
                        print("Unknown layer", layer)
                        return LRPPassLayer()
                    else:
                        raise NotImplementedError("The network contains layers that"
                                                  " are currently not supported {0:s}".format(str(layer)))

class LRPPassLayer(LRPLayer):
    #Add layer classes here to ignore them
    ALLOWED_PASS_LAYERS = []

class Module(LRPLayer, layer_class=torch.nn.Module):
    def relprop(self, module, relevance_in):
        for m in reversed(list(module.children())):
            print("Module running layer", m)
            relevance_in = LRPLayer.get(m).relprop_(m, relevance_in).detach()
            print("    relevence =", relevance_in)
        return relevance_in.detach()

class Container(LRPLayer, layer_class=torch.nn.Container):
    def relprop(self, module, relevance_in):
        with torch.no_grad():
            for m in reversed(module._modules.values()):
                print("Container running layer", m)
                relevance_in = LRPLayer.get(m).relprop_(m, relevance_in)
            return relevance_in

class Sequential(Container, layer_class=torch.nn.Sequential):
    pass

class ModuleList(Container, layer_class=torch.nn.ModuleList):
    pass

class ModuleDict(Container, layer_class=torch.nn.ModuleDict):
    pass

class LogSoftmax(LRPLayer, layer_class=torch.nn.LogSoftmax):
    def forward(self, m, in_tensor: torch.Tensor,
      out_tensor: torch.Tensor):
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in):
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
    def backward(self, module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        return (torch.clamp(grad_in[0], min=0.0),)

class Linear(LRPLayer, layer_class=torch.nn.Linear):
    AVAILABLE_METHODS = ["e-rule", "b-rule"]

    def forward(self, m, in_tensor: torch.Tensor,
      out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", list(out_tensor.size()))
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in):
        if LRPLayer.CURRENT_METHOD == "e-rule":
            with torch.no_grad():
                m.in_tensor = m.in_tensor.pow(LRPLayer.LRP_EXPONENT)
                w = m.weight.pow(LRPLayer.LRP_EXPONENT)
                norm = F.linear(m.in_tensor, w, bias=None)

                norm = norm + torch.sign(norm) * LRPLayer.EPS
                relevance_in[norm == 0] = 0
                norm[norm == 0] = 1
                relevance_out = F.linear(relevance_in / norm,
                                         w.t(), bias=None)
                relevance_out *= m.in_tensor
                del m.in_tensor, norm, w, relevance_in

        if LRPLayer.CURRENT_METHOD == "b-rule":
            with torch.no_grad():
                out_c, in_c = m.weight.size()
                w = m.weight.repeat((4, 1))
                # First and third channel repetition only contain the positive weights
                w[:out_c][w[:out_c] < 0] = 0
                w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
                # Second and fourth channel repetition with only the negative weights
                w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
                w[-out_c:][w[-out_c:] > 0] = 0

                # Repeat across channel dimension (pytorch always has channels first)
                m.in_tensor = m.in_tensor.repeat((1, 4))
                m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
                m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
                m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0

                # Normalize such that the sum of the individual importance values
                # of the input neurons divided by the norm
                # yields 1 for an output neuron j if divided by norm (v_ij in paper).
                # Norm layer just sums the importance values of the inputs
                # contributing to output j for each j. This will then serve as the normalization
                # such that the contributions of the neurons sum to 1 in order to
                # properly split up the relevance of j amongst its roots.

                norm_shape = m.out_shape
                norm_shape[1] *= 4
                norm = torch.zeros(norm_shape).to(self.innvestigator.device)

                for i in range(4):
                    norm[:, out_c * i:(i + 1) * out_c] = F.linear(
                        m.in_tensor[:, in_c * i:(i + 1) * in_c], w[out_c * i:(i + 1) * out_c], bias=None)

                # Double number of output channels for positive and negative norm per
                # channel.
                norm_shape[1] = norm_shape[1] // 2
                new_norm = torch.zeros(norm_shape).to(self.innvestigator.device)
                new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
                new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
                norm = new_norm

                # Some 'rare' neurons only receive either
                # only positive or only negative inputs.
                # Conservation of relevance does not hold, if we also
                # rescale those neurons by (1+beta) or -beta.
                # Therefore, catch those first and scale norm by
                # the according value, such that it cancels in the fraction.

                # First, however, avoid NaNs.
                mask = norm == 0
                # Set the norm to anything non-zero, e.g. 1.
                # The actual inputs are zero at this point anyways, that
                # is why norm is zero in the first place.
                norm[mask] = 1
                # The norm in the b-rule has shape (N, 2*out_c, *spatial_dims).
                # The first out_c block corresponds to the positive norms,
                # the second out_c block corresponds to the negative norms.
                # We find the rare neurons by choosing those nodes per channel
                # in which either the positive norm ([:, :out_c]) is zero, or
                # the negative norm ([:, :out_c]) is zero.
                rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

                # Also, catch new possibilities for norm == zero to avoid NaN..
                # The actual value of norm again does not really matter, since
                # the pre-factor will be zero in this case.

                norm[:, :out_c][rare_neurons] *= 1 if self.innvestigator.beta == -1 else 1 + self.innvestigator.beta
                norm[:, out_c:][rare_neurons] *= 1 if self.innvestigator.beta == 0 else -self.innvestigator.beta
                # Add stabilizer term to norm to avoid numerical instabilities.
                norm += self.eps * torch.sign(norm)
                input_relevance = relevance_in.squeeze(dim=-1).repeat(1, 4)
                input_relevance[:, :2*out_c] *= (1+self.beta)/norm[:, :out_c].repeat(1, 2)
                input_relevance[:, 2*out_c:] *= -self.beta/norm[:, out_c:].repeat(1, 2)
                inv_w = w.t()
                relevance_out = torch.zeros_like(m.in_tensor)
                for i in range(4):
                    relevance_out[:, i*in_c:(i+1)*in_c] = F.linear(
                        input_relevance[:, i*out_c:(i+1)*out_c],
                        weight=inv_w[:, i*out_c:(i+1)*out_c], bias=None)

                relevance_out *= m.in_tensor

                relevance_out = sum([relevance_out[:, i*in_c:(i+1)*in_c] for i in range(4)])

                del sum_weights, input_relevance, norm, rare_neurons, \
                    mask, new_norm, m.in_tensor, w, inv_w

        for attr in ("in_tensor", "out_shape"):
            if hasattr(m, attr):
                delattr(m, attr)
        return relevance_out.detach()

class _MaxPoolNd(LRPLayer):
    invert_pool = None

    def forward(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):
        # Save the return indices value to make sure
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "indices", indices)
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())

        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, layer_instance, relevance_in):
        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        with torch.no_grad():
            relevance_in = relevance_in.view(layer_instance.out_shape)

            inverted = invert_pool(relevance_in, layer_instance.indices,
                                   layer_instance.kernel_size, layer_instance.stride,
                                   layer_instance.padding, output_size=layer_instance.in_shape)
            del layer_instance.indices

        return inverted.detach()

class MaxPool1d(_MaxPoolNd, layer_class=torch.nn.MaxPool1d):
    invert_pool = F.max_unpool1d

class MaxPool2d(_MaxPoolNd, layer_class=torch.nn.MaxPool1d):
    invert_pool = F.max_unpool2d

class MaxPool3d(_MaxPoolNd, layer_class=torch.nn.MaxPool1d):
    invert_pool = F.max_unpool3d

class _ConvNd(LRPLayer):
    AVAILABLE_METHODS = ["e-rule", "b-rule"]

    def forward(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        setattr(m, 'out_tensor', out_tensor.features.copy())
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in):

        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        #relevance_in = self.reshape(m, relevance_in)

        if LRPLayer.CURRENT_METHOD == "e-rule":
            with torch.no_grad():
                m.in_tensor = m.in_tensor.pow(LRPLayer.LRP_EXPONENT).detach()
                w = m.weight.pow(LRPLayer.LRP_EXPONENT).detach()
                norm = self.conv_nd(
                    m,
                    m.in_tensor,
                    w,
                    bias=None)
                norm = norm + torch.sign(norm) * LRPLayer.EPS

                norm, relevance_in = self.reshape(norm, relevance_in)

                relevance_in[norm == 0] = 0
                norm[norm == 0] = 1
                x = relevance_in/norm
                relevance_out = self.inv_conv_nd(
                    m,
                    relevance_in/norm,
                    w,
                    bias=None)
                relevance_out *= m.in_tensor
                del m.in_tensor, norm, w

        elif LRPLayer.CURRENT_METHOD == "b-rule":
            with torch.no_grad():
                w = m.weight

                out_c, in_c = m.out_channels, m.in_channels
                repeats = np.array(np.ones_like(w.size()).flatten(), dtype=int)
                repeats[0] *= 4
                w = w.repeat(tuple(repeats))
                # First and third channel repetition only contain the positive weights
                w[:out_c][w[:out_c] < 0] = 0
                w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
                # Second and fourth channel repetition with only the negative weights
                w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
                w[-out_c:][w[-out_c:] > 0] = 0
                repeats = np.array(np.ones_like(m.in_tensor.size()).flatten(), dtype=int)
                repeats[1] *= 4
                # Repeat across channel dimension (pytorch always has channels first)
                m.in_tensor = m.in_tensor.repeat(tuple(repeats))
                m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
                m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
                m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0
                groups = 4

                # Normalize such that the sum of the individual importance values
                # of the input neurons divided by the norm
                # yields 1 for an output neuron j if divided by norm (v_ij in paper).
                # Norm layer just sums the importance values of the inputs
                # contributing to output j for each j. This will then serve as the normalization
                # such that the contributions of the neurons sum to 1 in order to
                # properly split up the relevance of j amongst its roots.
                norm = self.conv_nd(m, m.in_tensor, w, bias=None, scale_groups=groups)
                # Double number of output channels for positive and negative norm per
                # channel. Using list with out_tensor.size() allows for ND generalization
                new_shape = m.out_shape
                new_shape[1] *= 2
                new_norm = torch.zeros(new_shape).to(self.device)
                new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
                new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
                norm = new_norm
                # Some 'rare' neurons only receive either
                # only positive or only negative inputs.
                # Conservation of relevance does not hold, if we also
                # rescale those neurons by (1+beta) or -beta.
                # Therefore, catch those first and scale norm by
                # the according value, such that it cancels in the fraction.

                # First, however, avoid NaNs.
                mask = norm == 0
                # Set the norm to anything non-zero, e.g. 1.
                # The actual inputs are zero at this point anyways, that
                # is why norm is zero in the first place.
                norm[mask] = 1
                # The norm in the b-rule has shape (N, 2*out_c, *spatial_dims).
                # The first out_c block corresponds to the positive norms,
                # the second out_c block corresponds to the negative norms.
                # We find the rare neurons by choosing those nodes per channel
                # in which either the positive norm ([:, :out_c]) is zero, or
                # the negative norm ([:, :out_c]) is zero.
                rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

                # Also, catch new possibilities for norm == zero to avoid NaN..
                # The actual value of norm again does not really matter, since
                # the pre-factor will be zero in this case.

                norm[:, :out_c][rare_neurons] *= 1 if self.innvestigator.beta == -1 else 1 + self.innvestigator.beta
                norm[:, out_c:][rare_neurons] *= 1 if self.innvestigator.beta == 0 else -self.innvestigator.beta
                # Add stabilizer term to norm to avoid numerical instabilities.
                norm += LRPLayer.EPS * torch.sign(norm)
                spatial_dims = [1] * len(relevance_in.size()[2:])

                input_relevance = relevance_in.repeat(1, 4, *spatial_dims)
                input_relevance[:, :2*out_c] *= (1+self.innvestigator.beta)/norm[:, :out_c].repeat(1, 2, *spatial_dims)
                input_relevance[:, 2*out_c:] *= -self.innvestigator.beta/norm[:, out_c:].repeat(1, 2, *spatial_dims)
                # Each of the positive / negative entries needs its own
                # convolution. TODO: Can this be done in groups, too?

                relevance_out = torch.zeros_like(m.in_tensor)
                # Weird code to make up for loss of size due to stride
                tmp_result = result = None
                for i in range(4):
                    tmp_result = self.inv_conv_nd(
                        m,
                        input_relevance[:, i*out_c:(i+1)*out_c],
                        w[i*out_c:(i+1)*out_c],
                        bias=None)
                    result = torch.zeros_like(relevance_out[:, i*in_c:(i+1)*in_c])
                    tmp_size = tmp_result.size()
                    slice_list = [slice(0, l) for l in tmp_size]
                    result[slice_list] += tmp_result
                    relevance_out[:, i*in_c:(i+1)*in_c] = result
                relevance_out *= m.in_tensor

                sum_weights = torch.zeros([in_c, in_c * 4, *spatial_dims]).to(self.device)
                for i in range(m.in_channels):
                    sum_weights[i, i::in_c] = 1
                relevance_out = self.conv_nd(m, relevance_out, sum_weights, bias=None)

                del sum_weights, m.in_tensor, result, mask, rare_neurons, norm, \
                    new_norm, input_relevance, tmp_result, w
        for attr in ("in_tensor", "out_shape", 'out_tensor'):
            if hasattr(m, attr):
                if hasattr(getattr(m, attr), "detach"):
                    getattr(m, attr).detach()
                delattr(m, attr)

        self.clean(m)
        return relevance_out.detach()

    def clean(self, m):
        return

    def reshape(self, in_tensor, relevance_in):
        print("resize", m, m.out_shape, relevance_in.size())
        return in_tensor.size(), relevance_in.view(in_tensor.size())

    def conv_nd(self, layer, in_tensor, weight, bias=None, scale_groups=1):
        raise NotImplementedError

    def inv_conv_nd(self, layer, in_tensor, weight, bias=None,):
        raise NotImplementedError

class Conv1d(_ConvNd, layer_class=torch.nn.Conv1d):
    def conv_nd(self, layer, in_tensor, weight, bias=None, scale_groups=1):
        return F.conv1d(in_tensor, weight, bias=bias, stride=layer.stride,
            padding=layer.padding, dilation=layer.dilation,
            groups=scale_groups*layer.groups)

    def inv_conv_nd(self, layer, in_tensor, weight, bias=None):
        return F.conv_transpose1d(in_tensor, weight, bias=bias,
            stride=layer.stride, padding=layer.padding,
            output_padding=layer.output_padding, groups=layer.groups,
            dilation=layer.dilation)

class Conv2d(_ConvNd, layer_class=torch.nn.Conv2d):
    def conv_nd(self, layer, in_tensor, weight, bias=None, scale_groups=1):
        return F.conv2d(in_tensor, weight, bias=bias, stride=layer.stride,
            padding=layer.padding, dilation=layer.dilation,
            groups=scale_groups*layer.groups)

    def inv_conv_nd(self, layer, in_tensor, weight, bias=None):
        return F.conv_transpose2d(in_tensor, weight, bias=bias,
            stride=layer.stride, padding=layer.padding,
            output_padding=layer.output_padding, groups=layer.groups,
            dilation=layer.dilation)

class Conv3d(_ConvNd, layer_class=torch.nn.Conv3d):
    def conv_nd(self, layer, in_tensor, weight, bias=None, scale_groups=1):
        return F.conv3d(in_tensor, weight, bias=bias, stride=layer.stride,
            padding=layer.padding, dilation=layer.dilation,
            groups=scale_groups*layer.groups)

    def inv_conv_nd(self, layer, in_tensor, weight, bias=None):
        return F.conv_transpose3d(in_tensor, weight, bias=bias,
            stride=layer.stride, padding=layer.padding,
            output_padding=layer.output_padding, groups=layer.groups,
            dilation=layer.dilation)

class TransposeConvolutionND(LRPLayer):
    """https://github.com/etjoa003/medical_imaging/blob/master/isles2017/models/networks_LRP.py"""
    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0].detach())
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, 'out_tensor', out_tensor.detach())
        setattr(m, "in_spatial_size", in_tensor[0].spatial_size)
        setattr(m, "out_spatial_size", out_tensor.spatial_size)
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in):
        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.

        #Resize in and out to be same since dconvolution have different size
        Z, R = relprop_size_adjustment(m.out_tensor.features, m.in_tensor.features)

        m.weight = torch.max(0, m.weight)
        if len(m.bias.size()) == 0:
            m.bias = m.bias*0

        Z = m.out_tensor
        #Z, relevance_in = relprop_size_adjustment(Z, relevance_in)
        Z = Z + LRPLayer.EPS

        S = relevance_in / Z
        C = scn.convolution.ConvolutionFunction.apply(
            in_tensor,
            weight,
            torch.Tensor(m.nOut).zero_() if bias is None else bias,
            m.metadata,
            m.out_spatial_size,
            m.in_spatial_size,
            m.dimension,
            m.filter_size,
            m.filter_stride,
            )

        X, C = relprop_size_adjustment(m.out_tensor, C)
        R = X * C

        return R.detach()

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
