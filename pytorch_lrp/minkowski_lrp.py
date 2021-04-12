from typing import Union
from collections import Sequence

import torch
import numpy as np
import torch.nn.functional as F
from more_itertools import pairwise
from .modules import LRPLayer, LRPFunctionLayer, LRPPassLayer

import MinkowskiEngine as ME
from MinkowskiEngine.Common import convert_to_int_list, get_minkowski_function
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

def _get_coords_key(
        input: ME.SparseTensor,
        coords: Union[torch.IntTensor, ME.CoordsKey, ME.SparseTensor] = None,
        tensor_stride: Union[Sequence, np.ndarray, torch.IntTensor] = 1):
    r"""Process coords according to its type.
    """
    if coords is not None:
        assert isinstance(coords, (ME.CoordsKey, torch.IntTensor, ME.SparseTensor))
        if isinstance(coords, torch.IntTensor):
            coords_key = input.coords_man.create_coords_key(
                coords,
                tensor_stride=tensor_stride,
                force_creation=True,
                force_remap=True,
                allow_duplicate_coords=True)
        elif isinstance(coords, ME.SparseTensor):
            coords_key = coords.coords_key
        else:  # CoordsKey type due to the previous assertion
            coords_key = coords
    else:
        coords_key = ME.CoordsKey(input.D)
    return coords_key

class _(LRPPassLayer):
    #Add layer classes here to ignore them
    ALLOWED_PASS_LAYERS = [ME.MinkowskiSigmoid, ME.MinkowskiSoftmax,
                           ME.MinkowskiReLU, ME.MinkowskiPReLU, ME. MinkowskiELU,
                           ME.MinkowskiSELU, ME.MinkowskiCELU, ME.MinkowskiDropout,
                           ME.MinkowskiThreshold, ME.MinkowskiTanh,
                           ME.MinkowskiBatchNorm, ME.MinkowskiSyncBatchNorm,
                           ME.MinkowskiSyncBatchNorm, ME.MinkowskiInstanceNorm,
                           ME.MinkowskiStableInstanceNorm,
                           #ME.MinkowskiLinear, #MinkowskiLinear calls nn.Linear, which is already accounted for
                           BasicBlock, Bottleneck,
                           torch.nn.Linear]

def from_sparse_tensor(relevance):
    return relevance.F

def to_sparse_tensor(m, old_relevance, new_relevance):
    new = ME.SparseTensor(new_relevance, coords=m.in_coords) #,
        #coords_key=m.coords_key, coords_manager=m.coords_man)
    return new

class MinkwoskiLRPFunctionLayer(LRPFunctionLayer):
    TENSOR_CONV_TYPES = {ME.SparseTensor: {
        "from": from_sparse_tensor,
        "to": to_sparse_tensor}}

    # @classmethod
    # def relnormalize(cls, relevance):
    #     return ME.SparseTensor(super().relnormalize(relevance.F),
    #         coords_key=relevance.coords_key, coords_manager=relevance.coords_man)

    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        setattr(m, 'in_shape', in_tensor[0].F.size())
        setattr(m, "in_tensor", in_tensor[0].F)
        setattr(m, "in_coords", in_tensor[0].C)
        setattr(m, 'out_tensor', out_tensor.F)
        setattr(m, 'out_shape', out_tensor.F.size())
        setattr(m, "out_coords", out_tensor.C)

        try:
            region_type_, region_offset_, _ = \
                m.kernel_generator.get_kernel(in_tensor[0].tensor_stride, m.is_transpose)
        except torch.nn.modules.module.ModuleAttributeError:
            #Not a Conv or linear layer
            region_type_, region_offset_ = None, None

        out_coords_key = _get_coords_key(in_tensor[0], None)


        setattr(m, "tensor_stride", in_tensor[0].tensor_stride)
        setattr(m, "coords_key", in_tensor[0].coords_key)
        setattr(m, "out_coords_key", out_coords_key)
        setattr(m, "coords_man", in_tensor[0].coords_man)
        setattr(m, "region_type", region_type_)
        setattr(m, "region_offset", region_offset_)



        return None

    @staticmethod
    def clean(m, force=False):
        if force:
            del m.in_shape, m.in_tensor, m.out_tensor, m.out_shape, \
                m.tensor_stride, m.coords_key, m.out_coords_key, m.coords_man, \
                m.region_type, m.region_offset

class MinkLinear(MinkwoskiLRPFunctionLayer, layer_class=ME.MinkowskiLinear):
    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        MinkwoskiLRPFunctionLayer.forward_hook(m, in_tensor, out_tensor)
        setattr(m, "weight", m.linear.weight)

        return None

    @staticmethod
    def clean(m, force=False):
        MinkwoskiLRPFunctionLayer.clean(m, force)
        if force:
            del m.weight

    @staticmethod
    def forward_pass(m, tensor_in, weight):
        new = F.linear(tensor_in, weight, bias=None).detach()
        return new

    @staticmethod
    def backward_pass(m, tensor, weight):
        new = F.linear(tensor, weight.t(), bias=None).detach()
        return new

# from MinkowskiPooling import (
#     MinkowskiLocalPoolingFunction,
#     MinkowskiSumPooling,
#     MinkowskiAvgPooling,
#     MinkowskiMaxPooling,
#     MinkowskiLocalPoolingTransposeFunction,
#     MinkowskiPoolingTranspose,
#     MinkowskiGlobalPoolingFunction,
#     MinkowskiGlobalPooling,
#     MinkowskiGlobalSumPooling,
#     MinkowskiGlobalAvgPooling,
#     MinkowskiGlobalMaxPooling,
#     MinkowskiDirectMaxPoolingFunction,
# )

class MinkGlobalPooling(MinkwoskiLRPFunctionLayer, layer_class=ME.MinkowskiGlobalPooling):
    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        MinkwoskiLRPFunctionLayer.forward_hook(m, in_tensor, out_tensor)
        setattr(m, "weight", torch.Tensor([1.0]))

        return None

    @staticmethod
    def clean(m, force=False):
        MinkwoskiLRPFunctionLayer.clean(m, force)
        if force:
            del m.weight

    @staticmethod
    def forward_pass(m, in_tensor, weight, bias=None):
        with torch.no_grad():
            input_features = in_tensor.contiguous()

            fw_fn = get_minkowski_function('GlobalPoolingForward', input_features)
            out_feat, num_nonzero = fw_fn(input_features,
                                          m.coords_key.CPPCoordsKey,
                                          m.out_coords_key.CPPCoordsKey,
                                          m.coords_man.CPPCoordsManager,
                                          m.average, m.mode.value)

            setattr(m, "num_nonzero", num_nonzero)

        return out_feat.detach()

    @staticmethod
    def backward_pass(m, tensor, weight):
        with torch.no_grad():
            grad_out_feat = tensor.contiguous()

            bw_fn = get_minkowski_function('GlobalPoolingBackward', grad_out_feat)
            grad_in_feat = bw_fn(m.in_tensor, grad_out_feat, m.num_nonzero,
                                 m.coords_key.CPPCoordsKey,
                                 m.out_coords_key.CPPCoordsKey,
                                 m.coords_man.CPPCoordsManager, m.average)

            return grad_in_feat.detach()

class MinkConvolution(MinkwoskiLRPFunctionLayer, layer_class=ME.MinkowskiConvolution):
    NORMALIZE_AFTER = True

    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        MinkwoskiLRPFunctionLayer.forward_hook(m, in_tensor, out_tensor)
        setattr(m, "weight", m.kernel)

        return None

    @staticmethod
    def clean(m, force=False):
        MinkwoskiLRPFunctionLayer.clean(m, force)
        if force:
            del m.weight

    @staticmethod
    def forward_pass(m, in_tensor, weight, bias=None, scale_groups=1):
        with torch.no_grad():
            input_features = in_tensor

            if not input_features.is_contiguous():
                input_features = input_features.contiguous()

            D = m.coords_key.D
            out_feat = input_features.new()

            fw_fn = get_minkowski_function('ConvolutionForward',
                                           input_features)
            fw_fn(input_features, out_feat, weight,
                  convert_to_int_list(m.tensor_stride, D),
                  convert_to_int_list(m.stride, D),
                  convert_to_int_list(m.kernel_size, D),
                  convert_to_int_list(m.dilation, D), m.region_type, m.region_offset,
                  m.coords_key.CPPCoordsKey, m.out_coords_key.CPPCoordsKey,
                  m.coords_man.CPPCoordsManager)

            out_feat.detach()

            return out_feat

    @staticmethod
    def backward_pass(m, in_tensor, weight, bias=None):
        grad_out_feat = in_tensor
        with torch.no_grad():
            if not grad_out_feat.is_contiguous():
                grad_out_feat = grad_out_feat.contiguous()

            grad_in_feat = grad_out_feat.new()
            grad_kernel = grad_out_feat.new()
            D = m.coords_key.D
            bw_fn = get_minkowski_function('ConvolutionBackward', grad_out_feat)
            bw_fn(m.in_tensor, grad_in_feat, grad_out_feat, weight, grad_kernel,
                  convert_to_int_list(m.tensor_stride, D),
                  convert_to_int_list(m.stride, D),
                  convert_to_int_list(m.kernel_size, D),
                  convert_to_int_list(m.dilation, D), m.region_type,
                  m.coords_key.CPPCoordsKey, m.out_coords_key.CPPCoordsKey,
                  m.coords_man.CPPCoordsManager)

            grad_in_feat.detach()

            assert grad_in_feat.size() == m.in_shape, (grad_in_feat.size(), m.in_shape(), m.out_shape, m.in_coords.size())

            return grad_in_feat

class  MinkConvolutionTranspose(MinkwoskiLRPFunctionLayer, layer_class=ME.MinkowskiConvolutionTranspose):
    """https://github.com/etjoa003/medical_imaging/blob/master/isles2017/models/networks_LRP.py"""
    @staticmethod
    def forward_hook(m, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        MinkwoskiLRPFunctionLayer.forward_hook(m, in_tensor, out_tensor)
        setattr(m, "weight", m.kernel)

        return None

    @staticmethod
    def clean(m, force=False):
        MinkwoskiLRPFunctionLayer.clean(m, force)
        if force:
            del m.weight

    @staticmethod
    def forward_pass(m, in_tensor, weight, bias=None, scale_groups=1):
        with torch.no_grad():
            input_features = in_tensor

            if not input_features.is_contiguous():
                input_features = input_features.contiguous()

            D = m.coords_key.D
            out_feat = input_features.new()

            fw_fn = get_minkowski_function('ConvolutionTransposeForward',
                                           input_features)
            fw_fn(input_features, out_feat, weight,
                  convert_to_int_list(m.tensor_stride, D),
                  convert_to_int_list(m.stride, D),
                  convert_to_int_list(m.kernel_size, D),
                  convert_to_int_list(m.dilation, D), m.region_type, m.region_offset,
                  m.coords_key.CPPCoordsKey, m.out_coords_key.CPPCoordsKey,
                  m.coords_man.CPPCoordsManager, m.generate_new_coords)

            out_feat.detach()

            return out_feat

    @staticmethod
    def backward_pass(m, in_tensor, weight, bias=None):
        grad_out_feat = in_tensor
        with torch.no_grad():
            if not grad_out_feat.is_contiguous():
                grad_out_feat = grad_out_feat.contiguous()

            grad_in_feat = grad_out_feat.new()
            grad_kernel = grad_out_feat.new()
            D = m.coords_key.D
            bw_fn = get_minkowski_function('ConvolutionTransposeBackward', grad_out_feat)
            bw_fn(m.in_tensor, grad_in_feat, grad_out_feat, weight, grad_kernel,
                  convert_to_int_list(m.tensor_stride, D),
                  convert_to_int_list(m.stride, D),
                  convert_to_int_list(m.kernel_size, D),
                  convert_to_int_list(m.dilation, D), m.region_type,
                  m.coords_key.CPPCoordsKey, m.out_coords_key.CPPCoordsKey,
                  m.coords_man.CPPCoordsManager)

            grad_in_feat.detach()

            assert grad_in_feat.size() == m.in_shape, (grad_in_feat.size(), m.in_shape, m.in_coords.size())

            return grad_in_feat
