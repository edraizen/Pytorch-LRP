import torch
import numpy as np
import torch.nn.functional as F
from more_itertools import pairwise
from .modules import LRPLayer, LRPPassLayer, Container, _ConvNd
from .modules import ReLU as _ReLU

import sparseconvnet as scn
from sparseconvnet.metadata import Metadata

class SparseConvNetFeatureTensor(scn.SparseConvNetTensor):
    """Backwards hook require output to be a torch.Tensor, if it's not, it will
    try to get the 0th element"""
    def __getitem__(self, key):
        return self.features

class _(LRPPassLayer):
    #Add layer classes here to ignore them
    ALLOWED_PASS_LAYERS = [scn.Tanh, scn.Sigmoid, scn.ELU, scn.SELU,
                           scn.BatchNormELU, scn.BatchNormalization,
                           scn.Dropout, scn.BatchwiseDropout, scn.Identity,
                           scn.OutputLayer]

class InputLayer(LRPLayer, layer_class=scn.InputLayer):
    MIN = [0]
    MAX = [1]
    def forward(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        setattr(m, 'in_shape', in_tensor[0][1].size())
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_tensor', out_tensor.features)
        return LRPLayer.forward(self, m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in):
        """Z-beta rule from https://github.com/albermax/innvestigate/blob/master/innvestigate/analyzer/relevance_based/relevance_rule.py"""
        with torch.no_grad():
            if len(self.MIN) == 1:
                min_ = self.MIN*m.in_shape[1]
            elif len(self.MIN) == m.in_shape:
                min_ = self.MIN
            else:
                raise ValueError("InputLayer MIN must be of length 1 of size of features")

            if len(self.MAX) == 1:
                max_ = self.MIN*m.in_shape[1]
            elif len(self.MIN) == m.in_shape:
                min_ = self.MAX
            else:
                raise ValueError("InputLayer MAX must be of length 1 of size of features")

            print("Run input")

            low = torch.FloatTensor(min_).repeat(m.in_shape[0], 1).to(m.in_tensor[1].device)
            high = torch.FloatTensor(max_).repeat(m.in_shape[0], 1).to(m.in_tensor[1].device)


            low_forward = self.forward_pass(m, [m.in_tensor[0], low])
            high_forward = self.forward_pass(m, [m.in_tensor[0], high])
            print("high_forward", high_forward)

            Z = m.out_tensor-(low_forward+high_forward)

            S = relevance_in/(Z+self.EPS)

            tmpA = m.in_tensor[1]*self.backward_pass(m, m.in_tensor[1]+m.out_tensor+S)
            tmpB = low*self.backward_pass(m, self.in_tensor+low_forward+S)
            tmpC = high*self.backward_pass(m, self.in_tensor+high_forward+S)

            R = tmpA-(tmpB+tmpC)

            print("relout", R)

            del max_
            del min_
            del low
            del high
            del low_forward
            del high_forward
            del Z
            del S
            del tmpA
            del tmpB
            del tmpC
            del m.in_shape
            del m.in_tensor
            del m.out_tensor

            return R.detach()

    def forward_pass(self, m, in_tensor):
        return scn.ioLayers.InputLayerFunction.apply(
            m.dimension,
            Metadata(m.dimension),
            m.spatial_size,
            in_tensor[0],
            in_tensor[1],
            0 if len(in_tensor) == 2 else in_tensor[2],
            m.mode
        ).detach()

    def backward_pass(self, m, tensor):
        grad_input = tensor.new()
        scn.SCN.InputLayer_updateGradInput(
            Metadata(m.dimension),
            grad_input,
            tensor.contiguous())
        return grad_input.detach()


class ReLU(_ReLU, layer_class=scn.ReLU):
    def forward(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        if isinstance(out_tensor, scn.SparseConvNetTensor):
            out_tensor = SparseConvNetFeatureTensor(out_tensor.features,
                out_tensor.metadata, out_tensor.spatial_size)
        return out_tensor

class LeakyReLU(ReLU, layer_class=scn.LeakyReLU):
    pass

class BatchNormReLU(ReLU, layer_class=scn.BatchNormReLU):
    pass

class BatchNormLeakyReLU(ReLU, layer_class=scn.BatchNormLeakyReLU):
    pass

class MeanOnlyBNLeakyReLU(ReLU, layer_class=scn.MeanOnlyBNLeakyReLU):
    pass

class MaxPooling(LRPLayer, layer_class=scn.MaxPooling):
    def forward(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):
        # Save the return indices value to make sure
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "indices", indices)
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, 'in_shape', in_tensor[0].size())
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in):
        with torch.no_grad():
            # In case the output had been reshaped for a linear layer,
            # make sure the relevance is put into the same shape as before.
            relevance_in = relevance_in.view(m.out_shape).detach()

            inverted = scn.unPooling.UnPoolingFunction(relevance_in, m.metadata,
                m.spatial_size, m.in_shape, m.dimension, m.pool_size,
                m.pool_stride, m.nFeaturesToDrop)

            del relevance_in

        for attr in ("indices", "out_shape", 'in_shape'):
            if hasattr(m, attr):
                delattr(m, attr)

        return inverted.detach()

class Convolution(_ConvNd, layer_class=scn.Convolution):
    """Dense Convolutions with size 3 or 2, stride 2 can be conveterd todo
    scn.Convolutions"""
    NORMALIZE_BEFORE = True

    def forward(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        setattr(m, 'in_shape', in_tensor[0].features.size())
        setattr(m, "in_tensor", in_tensor[0].features)
        setattr(m, 'out_tensor', out_tensor.features)
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, "metadata", in_tensor[0].metadata)
        setattr(m, "in_spatial_size", in_tensor[0].spatial_size)
        setattr(m, "out_spatial_size", out_tensor.spatial_size)
        return LRPLayer.forward(self, m, in_tensor, out_tensor)

    def reshape(self, in_tensor, relevance_in):
        Z, R = sparse_relprop_size_adjustment(in_tensor, relevance_in)
        return Z, R

    def clean(self, m):
        for attr in ("in_shape", "in_tensor", "out_shape", 'metadata', "in_spatial_size"
          "out_spatial_size"):
            if hasattr(m, attr):
                delattr(m, attr)

    def conv_nd(self, m, in_tensor, weight, bias=None, scale_groups=1):
        with torch.no_grad():
            return scn.convolution.ConvolutionFunction.apply(
                in_tensor,
                weight,
                torch.Tensor(m.nOut).zero_() if bias is None else bias,
                m.metadata,
                m.in_spatial_size,
                m.out_spatial_size,
                m.dimension,
                m.filter_size,
                m.filter_stride,
                ) #groups=scale_groups * m.groups)

    def inv_conv_nd(self, m, in_tensor, weight, bias=None):
        with torch.no_grad():
            grad_input = in_tensor.new()
            grad_weight = torch.zeros_like(weight).detach()
            scn.SCN.Convolution_backward(
                m.in_spatial_size,
                m.out_spatial_size,
                m.filter_size,
                m.filter_stride,
                m.metadata,
                m.in_tensor,
                grad_input,
                in_tensor.contiguous(),
                weight,
                grad_weight,
                torch.Tensor())
            del grad_weight

            # deconv = scn.deconvolution.DeconvolutionFunction.apply(
            #     in_tensor,
            #     weight.permute(0,1,3,2),
            #     torch.Tensor(m.nOut).zero_() if bias is None else bias,
            #     m.metadata,
            #     m.out_spatial_size,
            #     m.in_spatial_size,
            #     m.dimension,
            #     m.filter_size,
            #     m.filter_stride,
            #     ) #groups=m.groups)
            # print("Deconv vs conv backward", deconv, grad_input, torch.allclose(deconv, grad_input))
            #del deconv
        return grad_input.detach()

class SubmanifoldConvolution(_ConvNd, layer_class=scn.SubmanifoldConvolution):
    """Dense Convolutions with size 3, stride 1, padding 1 can be replace by
    scn.SubmanifoldConvolutions"""

    def forward(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        setattr(m, 'in_shape', in_tensor[0].features.size())
        setattr(m, "in_tensor", in_tensor[0].features)
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, 'out_tensor', out_tensor.features)
        setattr(m, "metadata", in_tensor[0].metadata)
        setattr(m, "in_spatial_size", in_tensor[0].spatial_size)
        setattr(m, "out_spatial_size", out_tensor.spatial_size)
        return LRPLayer.forward(self, m, in_tensor, out_tensor)

    def clean(self, m):
        for attr in ("in_shape", "in_tensor", "out_shape", "out_tensor",
          'metadata', "in_spatial_size", "out_spatial_size"):
            if hasattr(m, attr):
                if hasattr(getattr(m, attr), "detach"):
                    getattr(m, attr).detach()
                delattr(m, attr)

    def reshape(self, in_tensor, relevance_in):
        Z, R = sparse_relprop_size_adjustment(in_tensor, relevance_in)
        return Z, R

    def conv_nd(self, m, in_tensor, weight, bias=None, scale_groups=1):
        with torch.no_grad():
            return scn.submanifoldConvolution.SubmanifoldConvolutionFunction.apply(
                in_tensor,
                weight,
                torch.Tensor() if not bias else bias, #torch.Tensor(m.nOut).zero_()
                m.metadata,
                m.in_spatial_size,
                m.dimension,
                m.filter_size)
                #groups=scale_groups*m.groups)

    def inv_conv_nd(self, m, in_tensor, weight, bias=None, stride=1, padding=0,
      output_padding=0, groups=1, dilation=1):
        with torch.no_grad():
            grad_input = in_tensor.new()
            grad_weight = torch.zeros_like(weight)
            scn.SCN.SubmanifoldConvolution_backward(
                m.in_spatial_size,
                m.filter_size,
                m.metadata,
                m.in_tensor,
                grad_input,
                in_tensor.contiguous(),
                weight,
                grad_weight,
                torch.Tensor())
            del grad_weight
        return grad_input.detach()

        # grad_input = grad_output.new()
        #
        # grad_bias = torch.zeros_like(bias)
        #from sparseconvnet.utils import toLongTensor

        # filter_stride = toLongTensor(m.dimension, self.filter_stride)
        #
        # #Flip input/output sizes
        # in_spatial_size = m.out_spatial_size
        # out_spatial_size = \
        #     (m.in_spatial_size - 1) * filter_stride + m.filter_size
        # # return scn.deconvolution.DeconvolutionFunction.apply(
        # #     in_tensor,
        # #     weight,
        # #     torch.Tensor() if not bias else bias,
        # #     m.metadata,
        # #     m.out_spatial_size, #m.input_spatial_size(m.out_spatial_size), #in_spatial_size,
        # #     m.in_spatial_size, #out_spatial_size,
        # #     m.dimension,
        # #     m.filter_size,
        # #     filter_stride
        # #     )

class Deconvolution(LRPLayer, layer_class=scn.Deconvolution):
    """https://github.com/etjoa003/medical_imaging/blob/master/isles2017/models/networks_LRP.py"""
    AVAILABLE_METHODS = ["e-rule", "0-rule", "g-rule"]
    NORMALIZE_AFTER = True

    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, 'out_tensor', out_tensor.features)
        setattr(m, 'metadata', out_tensor.metadata)
        setattr(m, "in_spatial_size", in_tensor[0].spatial_size)
        setattr(m, "out_spatial_size", out_tensor.spatial_size)
        return super().forward(m, in_tensor, out_tensor)

    def clean(self, m):
        for attr in ("in_shape", "out_shape", "out_tensor", "in_spatial_size",
          "out_spatial_size", "metadata"):
            if hasattr(m, attr):
                if hasattr(getattr(m, attr), "detach"):
                    getattr(m, attr).detach()
                delattr(m, attr)

    def relprop(self, m, relevance_in):
        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        if LRPLayer.CURRENT_METHOD not in ["e-rule", "0-rule", "g-rule"]:
            LRPLayer.CURRENT_METHOD  = "e-rule"

        with torch.no_grad():
             #torch.max(0, m.weight)

            weight = F.relu(m.weight).detach()
            if LRPLayer.CURRENT_METHOD == "g-rule":
                weight = (LRPLayer.GAMMA*weight).detach()
                Z = DeconvolutionFunction.apply(
                    m.in_tensor.features,
                    weight,
                    torch.Tensor(),
                    m.metadata,
                    m.in_spatial_size,
                    m.out_spatial_size,
                    m.dimension,
                    m.filter_size,
                    m.filter_stride)
            else:
                Z = m.out_tensor

            #Resize in and out to be same since deconvolution has different size
            Z, R = sparse_relprop_size_adjustment(Z, relevance_in)

            Z += self.EPSILON if LRPLayer.CURRENT_METHOD == "e-rule" else self.EPS

            S = R / Z
            grad_input = S.new()
            scn.SCN.Deconvolution_backward(
                m.in_spatial_size,
                m.out_spatial_size,
                m.filter_size,
                m.filter_stride,
                m.in_tensor.metadata,
                m.in_tensor.features,
                grad_input,
                S.contiguous(),
                weight,
                torch.zeros_like(weight),
                torch.Tensor())
            C = grad_input

            X, C = sparse_relprop_size_adjustment(m.in_tensor.features, C)
            R = X * C

            del Z
            del S
            del C
            del X
            del grad_input
            del weight
            self.clean(m)
        return R.detach()

def sparse_relprop_size_adjustment(Z,R):
    sZ, sR = Z.size(), R.size()
    if not np.all(sZ==sR):
        tempZ, tempR = get_zero_container(Z,R), get_zero_container(Z,R)
        tempZ[:sZ[0],:sZ[1]] = Z
        tempR[:sR[0],:sR[1]] = R

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
    return torch.zeros([max(sx, sy) for sx,sy in zip(list(x.size()),list(y.size()))])

class Sequential(Container, layer_class=scn.Sequential):
    pass

class ConcatTable(LRPLayer, layer_class=scn.ConcatTable):
    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        setattr(m, "in_tensors", in_tensor[0].features)
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, 'out_tensor', out_tensor)
        setattr(m, "in_spatial_sizes", [t.spatial_size for t in in_tensor[0]])
        setattr(m, "out_spatial_size", out_tensor.spatial_size)
        return super().forward(m, in_tensor, out_tensor)

    def clean(self, m):
        for attr in ("in_tensors", "out_shape", "out_tensor", "in_spatial_size",
          "out_spatial_size"):
            if hasattr(m, attr):
                if hasattr(getattr(m, attr), "detach"):
                    getattr(m, attr).detach()
                delattr(m, attr)

    def relprop(self, m, relevance_in):
        factor = 0.5

        def centre_crop_tensor(relevance_in, intended_shape):
            limits = []
            relavance_count = relevance_in.size()[1]
            correct_count = intended_shape[1]
            start = int(np.floor(np.abs(relavance_count-correct_count)/2) )
            stop = int(relavance_count-np.ceil(np.abs(relavance_count-correct_count)/2))
            assert stop-start == correct_count, (stop-start, correct_count)
            return relevance_in[:, start:stop]

        with torch.no_grad():
            relevances = [LRPLayer.get(m).relprop_(m, r).detach() for r, m in \
                zip(relevance_in, m._modules.values())]

            if len(relevance_in) == 2: # and isinstance(m._modules['0'], scn.Identity):
                relevance_out = factor*relevances[0] + (1-factor)*centre_crop_tensor(
                    relevances[1], relevances[0].size())
            else:
                relevance_out, _ = torch.stack(relevances).max(dim=0)

            del relevances

        return relevance_out.detach()

class JoinTable(LRPLayer, layer_class=scn.JoinTable):
    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        sizes = torch.Tensor([0]+[t.features.size()[1] for t in in_tensor[0]]).cumsum(dim=0).int()
        setattr(m, "in_sizes", sizes)
        setattr(m, "types", [t.__class__ for t in in_tensor[0]])
        return super().forward(m, in_tensor, out_tensor)

    def clean(self, m):
        for attr in ("in_sizes", "types"):
            if hasattr(m, attr):
                delattr(m, attr)

    def relprop(self, m, relevance_in):
        with torch.no_grad():
            relevance_out = [relevance_in[:, s1:s2].detach() for s1, s2 in \
                pairwise(m.in_sizes)]

            if len(m.types)==2: # and isinstance(m.types[0], scn.Identity):
                #Ignore idenity portion and use second relavance with correct size
                relevance2 = self.relnormalize(relevance_out[1]).detach()
                relevance1 = self.relnormalize(relevance2[:, :relevance_out[0].size()[1]]).detach()
                relevance_out = [relevance1, relevance2]
            else:
                print("??JoinTable", len(m.types)==2, m.types[0])


        self.clean(m)
        return relevance_out

class AddTable(LRPLayer, layer_class=scn.AddTable):
    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        setattr(m, "n_in_tensors", len(in_tensor[0]))
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in, use_mean=True):
        with torch.no_grad():
            if use_mean:
                relevance_in = relevance_in/m.n_in_tensors
            relevance_out = [relevance_in.detach()]*m.n_in_tensors

            del relevance_in
            del m.n_in_tensors
        return relevance_out

class NetworkInNetwork(LRPLayer, layer_class=scn.NetworkInNetwork):
    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, 'out_tensor', out_tensor.features)
        setattr(m, "in_spatial_size", in_tensor[0].spatial_size)
        setattr(m, "out_spatial_size", out_tensor.spatial_size)
        return super().forward(m, in_tensor, out_tensor)

    def clean(self, m):
        for attr in ("in_tensor", "out_shape", "out_tensor", "in_spatial_size",
          "out_spatial_size"):
            if hasattr(m, attr):
                delattr(m, attr)

    def relprop(self, m, relevance_in):
        if LRPLayer.CURRENT_METHOD in ["e-rule", "0-rule", "g-rule"]:
            LRPLayer.CURRENT_METHOD = "e-rule"

        with torch.no_grad():
            weight = m.weight.detach()

            if LRPLayer.CURRENT_METHOD == "g-rule":
                weight += (LRPLayer.GAMMA*F.relu(w)).detach()

            #Resize in and out to be same since deconvolution has different size
            Z, R = sparse_relprop_size_adjustment(m.out_tensor, relevance_in)
            Z += self.EPSILON if LRPLayer.CURRENT_METHOD == "e-rule" else self.EPS

            S = R / Z
            grad_input = S.new()
            scn.SCN.NetworkInNetwork_updateGradInput(
                grad_input,
                S,
                weight)
            C = grad_input.detach()

            X, C = sparse_relprop_size_adjustment(m.in_tensor.features, C)
            R = X * C

            del Z
            del S
            del C
            del X
            del weight
            del grad_input
        self.clean(m)
        return R.detach()

"""
class Unet():
    def relprop_standard(self,R , relprop_config):
		R = self.final_relu.relprop(R);
        R, ss = self.relnormalize(R, relprop_config) #Linear => norm

        R = self.bn.relprop(R)
		R = self.convf.relprop(R)
        R, ss = self.relnormalize(R, relprop_config) #Either Sequential or COncat Table =>norm

		R = self.cblocks7.relprop(R)
		R = R[:,self.paramdict['cb1'][1][1]:,:,:,:] # For Join Table, only keep
        R, ss = self.relnormalize(R, relprop_config)

		h1 = R[:,:self.paramdict['cb1'][1][1],:,:,:]
        h1, ss = self.relnormalize(h1, relprop_config)

		R = self.deconv3.relprop(R)
        R, ss = self.relnormalize(R, relprop_config)

		R = self.cblocks6.relprop(R)
		R = R[:,self.paramdict['cb2'][1][1]:]
        R, ss = self.relnormalize(R, relprop_config)

		h2 = R[:,:self.paramdict['cb2'][1][1],:,:,:]
        h2, ss = self.relnormalize(h2, relprop_config)

		R = self.deconv2.relprop(R)
        R, ss = self.relnormalize(R, relprop_config)

		R = self.cblocks5.relprop(R)
		R = R[:,self.paramdict['cb3'][1][1]:]
        R, ss = self.relnormalize(R, relprop_config)

		h3 = R[:,:self.paramdict['cb3'][1][1],:,:,:]
        h3, ss = self.relnormalize(h3, relprop_config)

		R = self.deconv1.relprop(R)
        R, ss = self.relnormalize(R, relprop_config)

		R = self.cblocks4.relprop(R)
        R, ss = self.relnormalize(R, relprop_config)

		R = self.pool3.relprop(R)

		factor = relprop_config['UNet3D']['concat_factors'][0]
		R = factor*h3 + (1-factor)*centre_crop_tensor(R, h3.shape).to(this_device)
		R = self.cblocks3.relprop(R,verbose=0)
        R, ss = self.relnormalize(R, relprop_config)

		R = self.pool2.relprop(R)

		factor2 = relprop_config['UNet3D']['concat_factors'][1]
		R = factor2*h2 + (1-factor2)* centre_crop_tensor(R, h2.shape).to(this_device)
		R = self.cblocks2.relprop(R)
        R, ss = self.relnormalize(R, relprop_config)

		R = self.pool1.relprop(R)

		factor3 = relprop_config['UNet3D']['concat_factors'][2]
		R = factor3*h1 + (1-factor3)*centre_crop_tensor(R, h1.shape).to(this_device)
		R = self.cblocks1.relprop(R)
        R, ss = self.relnormalize(R, relprop_config)

		return R
"""
