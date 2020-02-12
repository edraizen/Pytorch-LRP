import torch
import numpy as np
import torch.nn.functional as F
from more_itertools import pairwise
from .modules import LRPLayer, LRPFunctionLayer, LRPPassLayer, Container
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

class SparseLRPLayer(LRPLayer):
    pass

class SparseLRPFunctionLayer(LRPFunctionLayer):
    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        setattr(m, 'in_shape', in_tensor[0].features.size())
        setattr(m, "in_tensor", in_tensor[0].features)
        setattr(m, 'out_tensor', out_tensor.features)
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, "metadata", in_tensor[0].metadata)
        setattr(m, "in_spatial_size", in_tensor[0].spatial_size)
        setattr(m, "out_spatial_size", out_tensor.spatial_size)
        return None

    @staticmethod
    def clean(m):
        del m.in_shape, m.in_tensor, m.out_tensor, m.out_shape, m.metadata, \
            m.in_spatial_size, m.out_spatial_size

    @staticmethod
    def reshape(in_tensor, relevance_in):
        Z, R = sparse_relprop_size_adjustment(in_tensor, relevance_in)
        del in_tensor, relevance_in
        return Z, R

class InputLayer(SparseLRPLayer, layer_class=scn.InputLayer):
    RULE = "ZBetaRule"

    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        setattr(m, 'in_tensor', in_tensor[0])
        setattr(m, 'in_shape', in_tensor[0][1].size())
        setattr(m, 'metadata', out_tensor.metadata)

    @staticmethod
    def clean(m):
        del m.in_tensor, m.in_shape, m.metadata

    @staticmethod
    def forward_pass(m, in_tensor):
        with torch.no_grad():
            return scn.ioLayers.InputLayerFunction.apply(
                m.dimension,
                Metadata(m.dimension),
                m.spatial_size,
                in_tensor[0].cpu().long(),
                in_tensor[1],
                0 if len(in_tensor) == 2 else in_tensor[2],
                m.mode
            ).detach()

    @staticmethod
    def backward_pass(m, tensor):
        with torch.no_grad():
            grad_input = tensor.new()
            scn.SCN.InputLayer_updateGradInput(
                m.metadata,
                grad_input,
                tensor.contiguous())
        return grad_input.detach()

class ReLU(_ReLU, layer_class=scn.ReLU):
    @staticmethod
    def forward(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
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

class MaxPooling(SparseLRPFunctionLayer, layer_class=scn.MaxPooling):
    @staticmethod
    def forward_pass(m, tensor, weight, bias=None):
        return scn.MaxPoolingFunction.apply(
            tensor,
            m.metadata,
            m.in_spatial_size,
            m.out_spatial_size,
            m.dimension,
            m.pool_size,
            m.pool_stride,
            m.nFeaturesToDrop).detach()

    @staticmethod
    def backward_pass(m, tensor, weight):
        return scn.unPooling.UnPoolingFunction(tensor, m.metadata,
                m.spatial_size, m.in_shape, m.dimension, m.pool_size,
                m.pool_stride, m.nFeaturesToDrop).detach()

class Convolution(SparseLRPFunctionLayer, layer_class=scn.Convolution):
    """Dense Convolutions with size 3 or 2, stride 2 can be conveterd todo
    scn.Convolutions"""
    NORMALIZE_BEFORE = True

    @staticmethod
    def forward_pass(m, in_tensor, weight, bias=None, scale_groups=1):
        with torch.no_grad():
            with torch.no_grad():
                output_features = in_tensor.new()
                grad_bias = torch.Tensor().detach()
                scn.SCN.Convolution_updateOutput(
                    m.in_spatial_size,
                    m.out_spatial_size,
                    m.filter_size,
                    m.filter_stride,
                    m.metadata,
                    in_tensor,
                    output_features,
                    weight,
                    grad_bias)
                del grad_bias
            return output_features.detach()

    @staticmethod
    def backward_pass(m, in_tensor, weight, bias=None):
        with torch.no_grad():
            grad_input = in_tensor.new()
            grad_weight = torch.zeros_like(weight).detach()
            grad_bias = torch.Tensor().detach()
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
                grad_bias)
            del grad_weight, grad_bias

        return grad_input.detach()

class SubmanifoldConvolution(SparseLRPFunctionLayer, layer_class=scn.SubmanifoldConvolution):
    """Dense Convolutions with size 3, stride 1, padding 1 can be replace by
    scn.SubmanifoldConvolutions"""

    @staticmethod
    def forward_pass(m, in_tensor, weight, bias=None, scale_groups=1):
        with torch.no_grad():
            output_features = in_tensor.new()
            grad_bias = torch.Tensor().detach()
            scn.SCN.SubmanifoldConvolution_updateOutput(
                m.in_spatial_size,
                m.filter_size,
                m.metadata,
                in_tensor,
                output_features,
                weight,
                grad_bias)
            del grad_bias
        return output_features.detach()

    @staticmethod
    def backward_pass(m, in_tensor, weight):
        with torch.no_grad():
            grad_input = in_tensor.new()
            grad_weight = torch.zeros_like(weight).detach()
            grad_bias = torch.Tensor().detach()
            input = in_tensor.contiguous().detach()
            scn.SCN.SubmanifoldConvolution_backward(
                m.in_spatial_size,
                m.filter_size,
                m.metadata,
                m.in_tensor,
                grad_input,
                input,
                weight,
                grad_weight,
                grad_bias
                )
            del grad_weight, grad_bias, input
        return grad_input.detach()

class Deconvolution(SparseLRPFunctionLayer, layer_class=scn.Deconvolution):
    """https://github.com/etjoa003/medical_imaging/blob/master/isles2017/models/networks_LRP.py"""
    NORMALIZE_AFTER = True

    @staticmethod
    def forward_pass(m, in_tensor, weight):
        with torch.no_grad():
            output_features = in_tensor.new()
            grad_bias = torch.Tensor().detach()
            scn.SCN.Deconvolution_updateOutput(
                m.in_spatial_size,
                m.out_spatial_size,
                m.filter_size,
                m.filter_stride,
                m.metadata,
                in_tensor,
                output_features,
                weight,
                grad_bias)
            del grad_bias
        return output_features.detach()

    @staticmethod
    def backward_pass(m, in_tensor, weight):
        with torch.no_grad():
            grad_input = in_tensor.new()
            grad_weight = torch.zeros_like(weight).detach()
            grad_bias = torch.Tensor().detach()
            scn.SCN.Deconvolution_backward(
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
                grad_bias)
            del grad_bias, grad_weight
        return grad_input.detach()

class NetworkInNetwork(SparseLRPFunctionLayer, layer_class=scn.NetworkInNetwork):
    @staticmethod
    def forward_pass(m, in_tensor, weight):
        with torch.no_grad():
            output_features = in_tensor.new()
            grad_bias = torch.Tensor().detach()
            scn.SCN.NetworkInNetwork_updateOutput(
                in_tensor,
                output_features,
                weight,
                grad_bias)
            del grad_bias
        return output_features.detach()

    @staticmethod
    def backward_pass(m, in_tensor, weight):
        with torch.no_grad():
            grad_input = in_tensor.new()
            scn.SCN.NetworkInNetwork_updateGradInput(
                grad_input,
                in_tensor,
                weight)
        return grad_input.detach()

def sparse_relprop_size_adjustment(Z,R):
    sZ, sR = Z.size(), R.size()
    if not np.all(sZ==sR):
        tempZ, tempR = get_zero_container(Z,R), get_zero_container(Z,R)
        print("tZ", sZ, tempZ.size())
        print("tR", sR, tempR.size())
        tempZ[:sZ[0],:sZ[1]] = Z
        tempR[:sR[0],:sR[1]] = R
        del Z, R
        return tempZ.detach(), tempR.detach()
    else:
        return Z.detach(), R.detach()

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

class ConcatTable(SparseLRPLayer, layer_class=scn.ConcatTable):
    @classmethod
    def relprop(cls, m, relevance_in):
        factor = 0.5
        nIn = len(relevance_in)

        def centre_crop_tensor(relevance_in, intended_shape):
            relavance_count = relevance_in.size()[1]
            correct_count = intended_shape[1]
            start = int(np.floor(np.abs(relavance_count-correct_count)/2) )
            stop = int(relavance_count-np.ceil(np.abs(relavance_count-correct_count)/2))
            assert stop-start == correct_count, (stop-start, correct_count)
            return relevance_in[:, start:stop]

        with torch.no_grad():
            relevances = [LRPLayer.get(m).relprop_(m, r).detach() for r, m in \
                zip(relevance_in, m._modules.values())]
            del relevance_in

            if nIn == 2: # and isinstance(m._modules['0'], scn.Identity):
                relevance_out = factor*relevances[0] + (1-factor)*centre_crop_tensor(
                    relevances[1], relevances[0].size())
            else:
                relevance_out, _ = torch.stack(relevances).max(dim=0)
                del _

            del relevances

        return relevance_out.detach()

class JoinTable(SparseLRPLayer, layer_class=scn.JoinTable):
    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        sizes = torch.Tensor([0]+[t.features.size()[1] for t in in_tensor[0]]).cumsum(dim=0).int()
        setattr(m, "in_sizes", sizes)
        setattr(m, "types", [t.__class__ for t in in_tensor[0]])
        del sizes

    @classmethod
    def relprop(cls, m, relevance_in):
        with torch.no_grad():
            relevance_out = [relevance_in[:, s1:s2].detach() for s1, s2 in \
                pairwise(m.in_sizes)]
            del relevance_in

            if len(m.types)==2: # and isinstance(m.types[0], scn.Identity):
                #Ignore idenity portion and use second relavance with correct size
                relevance2 = cls.relnormalize(relevance_out[1]).detach()
                relevance1 = cls.relnormalize(relevance2[:, :relevance_out[0].size()[1]]).detach()
                relevance_out = [relevance1, relevance2]
                del relevance1, relevance2
            else:
                print("??JoinTable", len(m.types)==2, m.types[0])

        del m.in_sizes, m.types
        return relevance_out

class AddTable(SparseLRPLayer, layer_class=scn.AddTable):
    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        setattr(m, "n_in_tensors", len(in_tensor[0]))

    @classmethod
    def relprop(cls, m, relevance_in, use_mean=True):
        with torch.no_grad():
            if use_mean:
                relevance_in = relevance_in/m.n_in_tensors
            relevance_out = [relevance_in.detach()]*m.n_in_tensors

            del relevance_in
            del m.n_in_tensors
        return relevance_out

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
