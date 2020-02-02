import torch
import numpy as np
import torch.nn.functional as F

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))

ALLOWED_LAYERS = []
ALLOWED_LAYERS_BY_NAME = {}
AVAILABLE_METHODS = set()
ALLOWED_PASS_LAYERS = [
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
    torch.nn.ReLU, torch.nn.ELU, Flatten,
    torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d,
    torch.nn.Softmax, torch.nn.LogSoftmax, torch.nn.Sigmoid]

class LRPLayer(object):
    AVAILABLE_METHODS = []

    def __init__(self, innvestigator, method):
        if len(self.AVAILABLE_METHODS) > 0:
            assert method in self.AVAILABLE_METHODS
        self.innvestigator = innvestigator
        self.method = method

    @classmethod
    def __init_subclass__(cls, layer_class=None, **kwds):
        super().__init_sublcass(*kwds)
        ALLOWED_LAYERS.append(cls)
        if layer_class is not None:
            layer_class = cls.__name__.rsplit(".", 1)[-1]
        ALLOWED_LAYERS_BY_NAME[layer_class] = cls
        AVAILABLE_METHODS |= set(cls.AVAILABLE_METHODS)

    def forward(self, m, in_tensor: torch.Tensor,
      out_tensor: torch.Tensor):
        self.innvestigator.module_list.append(m)
        return None

    def relprop(self, m, relevance_in):
        return relevance_in

class LRPPassLayer(LRPLayer):
    #Add layer classes here to ignore them
    ALLOWED_PASS_LAYERS = []

    @classmethod
    def __init_subclass__(cls, layer_class=None, **kwds):
        super().__init_sublcass(*kwds)
        if isinstance(cls.ALLOWED_PASS_LAYERS, (list, tuple)):
            ALLOWED_PASS_LAYERS += cls.ALLOWED_PASS_LAYERS

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
        if relevance.sum() < 0:
            relevance[relevance == 0] = -1e6
            relevance = relevance.exp()
        return relevance

class Linear(LRPLayer, layer_class=torch.nn.Linear):
    AVAILABLE_METHODS = ["e-rule", "b-rule"]

    def forward(self, m, in_tensor: torch.Tensor,
      out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", list(out_tensor.size()))
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in):
        if self.method == "e-rule":
            m.in_tensor = m.in_tensor.pow(self.innvestigator.p)
            w = m.weight.pow(self.innvestigator.p)
            norm = F.linear(m.in_tensor, w, bias=None)

            norm = norm + torch.sign(norm) * self.innvestigator.eps
            relevance_in[norm == 0] = 0
            norm[norm == 0] = 1
            relevance_out = F.linear(relevance_in / norm,
                                     w.t(), bias=None)
            relevance_out *= m.in_tensor
            del m.in_tensor, norm, w, relevance_in
            return relevance_out.detach()

        if self.method == "b-rule":
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
    inv_conv_nd = None
    conv_nd = None

    def forward(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in):

        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        relevance_in = relevance_in.view(m.out_shape)

        if self.method == "e-rule":
            with torch.no_grad():
                m.in_tensor = m.in_tensor.pow(self.innvestigator.p).detach()
                w = m.weight.pow(self.innvestigator.p).detach()
                norm = self.conv_nd(m.in_tensor, weight=w, bias=None,
                               stride=m.stride, padding=m.padding,
                               groups=m.groups)

                norm = norm + torch.sign(norm) * self.innvestigator.eps
                relevance_in[norm == 0] = 0
                norm[norm == 0] = 1
                relevance_out = self.inv_conv_nd(relevance_in/norm,
                                            weight=w, bias=None,
                                            padding=m.padding, stride=m.stride,
                                            groups=m.groups)
                relevance_out *= m.in_tensor
                del m.in_tensor, norm, w
                return relevance_out.detach()

        if self.method == "b-rule":
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
                norm = self.conv_nd(m.in_tensor, weight=w, bias=None, stride=m.stride,
                               padding=m.padding, dilation=m.dilation, groups=groups * m.groups)
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
                norm += self.innvestigator.eps * torch.sign(norm)
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
                        input_relevance[:, i*out_c:(i+1)*out_c],
                        weight=w[i*out_c:(i+1)*out_c],
                        bias=None, padding=m.padding, stride=m.stride,
                        groups=m.groups)
                    result = torch.zeros_like(relevance_out[:, i*in_c:(i+1)*in_c])
                    tmp_size = tmp_result.size()
                    slice_list = [slice(0, l) for l in tmp_size]
                    result[slice_list] += tmp_result
                    relevance_out[:, i*in_c:(i+1)*in_c] = result
                relevance_out *= m.in_tensor

                sum_weights = torch.zeros([in_c, in_c * 4, *spatial_dims]).to(self.device)
                for i in range(m.in_channels):
                    sum_weights[i, i::in_c] = 1
                relevance_out = self.conv_nd(relevance_out, weight=sum_weights, bias=None)

                del sum_weights, m.in_tensor, result, mask, rare_neurons, norm, \
                    new_norm, input_relevance, tmp_result, w

                return relevance_out.detach()

class Conv1d(_ConvNd, layer_class=torch.nn.Conv1d):
    conv_nd = F.conv1d
    inv_conv_nd = F.conv_transpose1d

class Conv2d(_ConvNd, layer_class=torch.nn.Conv2d):
    conv_nd = F.conv2d
    inv_conv_nd = F.conv_transpose2d

class Conv3d(_ConvNd, layer_class=torch.nn.Conv3d):
    conv_nd = F.conv3d
    inv_conv_nd = F.conv_transpose3d
