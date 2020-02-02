import torch
import numpy as np

from .modules import LRPLayer, LRPPassLayer

import sparseconvnet as scn

def optionalTensor(a, b):
    return getattr(a, b) if hasattr(a, b) else torch.Tensor()

class _(LRPPassLayer):
    #Add layer classes here to ignore them
    ALLOWED_PASS_LAYERS = [torch.nn.Softmax, torch.nn.LogSoftmax,
                           scn.Tanh, scn.Sigmoid, scn.ReLU, scn.LeakyReLU,
                           scn.ELU, scn.SELU, scn.BatchNormELU,
                           scn.BatchNormalization, scn.BatchNormReLU,
                           scn.BatchNormLeakyReLU, scn.MeanOnlyBNLeakyReLU,
                           scn.Dropout, scn.BatchwiseDropout,
                           scn.ConcatTable, scn.AddTable, scn.JoinTable,
                           scn.Identity, scn.NetworkInNetwork, scn.Sequential]

class MaxPool(LRPLayer, layer_class=scn.MaxPool):
    def forward(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):
        # Ignore unused for pylint
        _ = self

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

        inverted = scn.UnPoolingFunction(relevance_in, layer_instance.metadata,
            layer_instance.spatial_size, layer_instance.in_shape,
            layer_instance.dimension, layer_instance.pool_size,
            layer_instance.pool_stride, layer_instance.nFeaturesToDrop)

        return inverted

def Convolution(LRPLayer, layer_class=scn.Convolution):
    """Dense Convolutions with size 3 or 2, stride 2 can be conveterd todo
    scn.Convolutions"""
    conv_fn = scn.ConvolutionFunction
    inv_conv_fun = scn.DeconvolutionFunction

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
                if hasattr(m, "filter_stride"):
                    norm = self.conv_fn(relevance_in, m.weight,
                        optionalTensor(m, 'bias'), m.metadata,
                        m.spatial_size, m.in_shape, m.dimension,
                        m.filter_size, m.filter_stride, groups=m.groups)
                        filter_stride = m.filter_stride
                else:
                    #SubmanifoldConvolution
                    norm = self.conv_fn(relevance_in, m.weight,
                        optionalTensor(m, 'bias'), m.metadata,
                        m.spatial_size, m.in_shape, m.dimension,
                        m.filter_size, groups=m.groups)
                    filter_stride = 1

                norm = norm + torch.sign(norm) * self.eps
                relevance_in[norm == 0] = 0
                norm[norm == 0] = 1
                relevance_out = self.inv_conv_nd(relevance_in/norm, m.weight,
                    optionalTensor(self, 'bias'), relevance_in.metadata,
                    m.spatial_size, m.in_shape, m.dimension,
                    m.filter_size, filter_stride, groupe=m.groups)
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
                if hasattr(m, "filter_stride"):
                    norm = self.conv_fn(relevance_in, m.weight,
                        optionalTensor(m, 'bias'), m.metadata,
                        m.spatial_size, m.in_shape, m.dimension,
                        m.filter_size, m.filter_stride, groups=groups * m.groups)
                        filter_stride = m.filter_stride
                else:
                    #SubmanifoldConvolution
                    norm = self.conv_fn(relevance_in, m.weight,
                        optionalTensor(m, 'bias'), relevance_in.metadata,
                        m.spatial_size, m.in_shape, m.dimension,
                        m.filter_size, groups=groups * m.groups)
                    filter_stride = 1
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
                        w[i*out_c:(i+1)*out_c], #Weight
                        optionalTensor(m, 'bias'), m.metadata,
                        m.spatial_size, m.in_shape, m.dimension,
                        m.filter_size, filter_stride, groupe=m.groups)
                    result = torch.zeros_like(relevance_out[:, i*in_c:(i+1)*in_c])
                    tmp_size = tmp_result.size()
                    slice_list = [slice(0, l) for l in tmp_size]
                    result[slice_list] += tmp_result
                    relevance_out[:, i*in_c:(i+1)*in_c] = result
                relevance_out *= m.in_tensor

                sum_weights = torch.zeros([in_c, in_c * 4, *spatial_dims]).to(self.device)
                for i in range(m.in_channels):
                    sum_weights[i, i::in_c] = 1
                if hasattr(relevance_in, "filter_stride"):
                    norm = self.conv_nd(relevance_out, sum_weights,
                        optionalTensor(m, 'bias'), m.metadata,
                        m.spatial_size, m.in_shape, m.dimension,
                        0, 1, groups=m.groups)
                else:
                    norm = self.conv_nd(relevance_out, sum_weights,
                        optionalTensor(m, 'bias'), m.metadata,
                        m.spatial_size, m.in_shape, m.dimension,
                        0, groups=m.groups)

                del sum_weights, m.in_tensor, result, mask, rare_neurons, norm, \
                    new_norm, input_relevance, tmp_result, w

                return relevance_out.detach()

class SubmanifoldConvolution(LRPLayer, layer_class=scn.SubmanifoldConvolution):
    """Dense Convolutions with size 3, stride 1, padding 1 can be replace by
    scn.SubmanifoldConvolutions"""
    conv_fn = scn.SubmanifoldConvolutionFunction
    inv_conv_fn = scn.DeconvolutionFunction

class Deconvolution(LRPLayer, layer_class=scn.Deconvolution):
    """https://github.com/etjoa003/medical_imaging/blob/master/isles2017/models/networks_LRP.py"""
    buffer = 1e-9

    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        setattr(m, 'out_tensor', out_tensor)
        return super().forward(m, in_tensor, out_tensor)

	def relprop(self, m, relevance_in):
        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        relevance_in = relevance_in.view(m.out_shape)

		m.weight = torch.max(0, m.weight)
		if len(m.bias.size()) == 0:
            m.bias = m.bias*0

		Z = m.out_tensor
		#Z, relevance_in = relprop_size_adjustment(Z, relevance_in)
		Z = Z + self.buffer

		S = relevance_in / Z
		C = scn.Convolution(S, m.weight, optionalTensor(m, 'bias'),
            m.metadata, m.spatial_size, m.in_shape, m.dimension,
            m.filter_size, groups=m.groups)

		X, C = relprop_size_adjustment(m.out_tensor, C)
		R = X * C

		return R.detach()

def relprop_size_adjustment(Z,R):
	sZ, sR = Z.shape, R.shape
	if not np.all(sZ==sR):
		tempR, tempZ = get_zero_container(Z,R), get_zero_container(Z,R)
		tempR[:sR[0],:sR[1],:sR[2],:sR[3],:sR[4]] = R; # R = tempR
		tempZ[:sZ[0],:sZ[1],:sZ[2],:sZ[3],:sZ[4]] = Z; # Z = tempZ
	else: return Z, R
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
	s = []
	for sx,sy in zip(x.shape,y.shape):
		s.append(np.max([sx,sy]))
	return torch.zeros(s)
