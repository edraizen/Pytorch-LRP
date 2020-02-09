import torch
import numpy as np
import torch.nn.functional as F
from more_itertools import pairwise
from .modules import LRPLayer, LRPPassLayer, Container, _ConvNd

import sparseconvnet as scn
from sparseconvnet.utils import toLongTensor

def optionalTensor(a, b):
    return getattr(a, b) if hasattr(a, b) else torch.Tensor()

class _(LRPPassLayer):
    #Add layer classes here to ignore them
    ALLOWED_PASS_LAYERS = [scn.Tanh, scn.Sigmoid, scn.ReLU, scn.LeakyReLU,
                           scn.ELU, scn.SELU, scn.BatchNormELU,
                           scn.BatchNormalization, scn.BatchNormReLU,
                           scn.BatchNormLeakyReLU, scn.MeanOnlyBNLeakyReLU,
                           scn.Dropout, scn.BatchwiseDropout, scn.Identity,
                           scn.InputLayer, scn.OutputLayer]

class MaxPooling(LRPLayer, layer_class=scn.MaxPooling):
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
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, 'in_shape', in_tensor[0].size())
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, layer_instance, relevance_in):
        with torch.no_grad():
            # In case the output had been reshaped for a linear layer,
            # make sure the relevance is put into the same shape as before.
            relevance_in = relevance_in.view(layer_instance.out_shape).detach()

            inverted = scn.unPooling.UnPoolingFunction(relevance_in, layer_instance.metadata,
                layer_instance.spatial_size, layer_instance.in_shape,
                layer_instance.dimension, layer_instance.pool_size,
                layer_instance.pool_stride, layer_instance.nFeaturesToDrop)

            del relevance_in

        for attr in ("indices", "out_shape", 'in_shape'):
            if hasattr(m, attr):
                delattr(m, attr)

        return inverted.detach()

class Convolution(_ConvNd, layer_class=scn.Convolution):
    """Dense Convolutions with size 3 or 2, stride 2 can be conveterd todo
    scn.Convolutions"""
    def forward(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        setattr(m, 'in_shape', in_tensor[0].features.size())
        setattr(m, "in_tensor", in_tensor[0].features)
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
        assert m.groups == 1, "Grouped convolutions not implemented"
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
        print("conv-inv=deconv")
        print(in_tensor, in_tensor.size())
        print(m.in_spatial_size, m.out_spatial_size)
        print(weight, weight.size())
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

        return

class SubmanifoldConvolution(_ConvNd, layer_class=scn.SubmanifoldConvolution):
    """Dense Convolutions with size 3, stride 1, padding 1 can be replace by
    scn.SubmanifoldConvolutions"""
    #Set fake filter size for deconvolution
    filter_stride = 1

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
        print("in_tensor", in_tensor, in_tensor.size())
        print("relevance_in", relevance_in, relevance_in.size())
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
        print("sub-inv=Sub-back", m)
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
        #
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
    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', out_tensor.features.size())
        setattr(m, 'out_tensor', out_tensor.features)
        setattr(m, "in_spatial_size", in_tensor[0].spatial_size)
        setattr(m, "out_spatial_size", out_tensor.spatial_size)
        return super().forward(m, in_tensor, out_tensor)

    def clean(self, m):
        for attr in ("in_shape", "out_shape", "out_tensor", "in_spatial_size",
          "out_spatial_size"):
            if hasattr(m, attr):
                if hasattr(getattr(m, attr), "detach"):
                    getattr(m, attr).detach()
                delattr(m, attr)

    def relprop(self, m, relevance_in):
        # In case the output had been reshaped for a linear layer,
        # make sure the relevance is put into the same shape as before.
        with torch.no_grad():
            weight = F.relu(m.weight.detach()) #torch.max(0, m.weight)
            print("deconv-inv")
            if hasattr(m, "bias"):
                m.bias = m.bias*0

            #Resize in and out to be same since deconvolution has different size
            print("before resize", m.out_tensor.size(), relevance_in.size())
            Z, R = sparse_relprop_size_adjustment(m.out_tensor, relevance_in)
            Z = Z + LRPLayer.EPS
            print("after resize", Z.size(), R.size())

            print("weight", weight.size())
            print("size", m.filter_size)

            S = R / Z
            # C = scn.convolution.ConvolutionFunction.apply(
            #     S,
            #     weight,
            #     torch.Tensor(),
            #     m.out_tensor.metadata,
            #     m.out_spatial_size,
            #     m.in_spatial_size,
            #     m.dimension,
            #     m.filter_size,
            #     m.filter_stride,
            #     )
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

            print("before resize2", m.in_tensor.features.size(), C.size())
            X, C = sparse_relprop_size_adjustment(m.in_tensor.features, C)
            print("after resize2", X.size(), C.size())
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
    print(x.size(), list(x.size()))
    print(y.size(), list(y.size()))
    print([max(sx, sy) for sx,sy in zip(list(x.size()),list(y.size()))])
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
            relevance_in = [LRPLayer.get(m).relprop_(m, r).detach() for r, m in \
                zip(relevance_in, m._modules.values())]

            print("ConcatTable rel in", relevance_in)
            print("ConcatTable rel in", [r.size() for r in relevance_in])

            relevance_out, _ = torch.stack(relevance_in).max(dim=0)
            print("ConcatTable rel out", relevance_out.size())

            # if len(relevance_in)>1:
            #     #First tensor is usually from a skip connection, so only use
            #     #Relevances from the other layers. If multple, take an the mean
            #     relevance_out = torch.stack(relevance_in[1:]).mean(dim=0)
            #     print("ConcatTable RelOut", relevance_out, relevance_out.size())
            #     h = relevance_out[:, :relevance_in[0].size()[1]].detach()
            #     relevance_out = factor*h + (1-factor)*centre_crop_tensor(relevance_out,
            #         h.size())
            # else:
            #     relevance_out = relevance_in[0]
            #     h = None

            del relevance_in
            #del h

        return relevance_out.detach()

class JoinTable(LRPLayer, layer_class=scn.JoinTable):
    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        sizes = torch.Tensor([0]+[t.features.size()[1] for t in in_tensor[0]]).cumsum(dim=0).int()
        setattr(m, "in_tensors_sizes", sizes)
        return super().forward(m, in_tensor, out_tensor)

    def clean(self, m):
        for attr in ("in_tensors", "out_shape", "out_tensor", "in_spatial_size",
          "out_spatial_size"):
            if hasattr(m, attr):
                delattr(m, attr)

    def relprop(self, m, relevance_in):
        with torch.no_grad():
            for i, s in enumerate(m.in_tensors_sizes):
                print("==>({}): {}".format(i, s))
            relevance_out = [relevance_in[:, s1:s2].detach() for s1, s2 in \
                pairwise(m.in_tensors_sizes)]
            print("||==>({}): {}".format(i+1, [r.size() for r in relevance_out]))

        del m.in_tensors_sizes
        return relevance_out

class AddTable(LRPLayer, layer_class=scn.AddTable):
    def forward(self, m, in_tensor: torch.Tensor,
                          out_tensor: torch.Tensor):
        setattr(m, "n_in_tensors", len(in_tensor[0]))
        return super().forward(m, in_tensor, out_tensor)

    def relprop(self, m, relevance_in, use_mean=False):
        with torch.no_grad():
            if use_mean:
                relevance_in = relevance_in/m.n_in_tensors
            relevance_out = [relevance_in.detach()]*m.n_in_tensors
            print("AddTable relout", relevance_out)
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
        with torch.no_grad():
            weight = F.relu(m.weight).detach()

            #Resize in and out to be same since deconvolution has different size
            Z, R = sparse_relprop_size_adjustment(m.out_tensor, relevance_in)
            Z = Z + LRPLayer.EPS

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
