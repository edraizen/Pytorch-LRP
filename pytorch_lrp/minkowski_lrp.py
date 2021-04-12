from typing import Union
from collections import Sequence

import torch
import numpy as np
import torch.nn.functional as F
from more_itertools import pairwise
from .modules import LRPLayer, LRPFunctionLayer, LRPPassLayer

import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import _get_coordinate_map_key
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

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

        out_coordinate_map_key = _get_coordinate_map_key(input, coordinates)

        setattr(m, "tensor_stride", in_tensor[0].tensor_stride)
        setattr(m, "in_coordinate_map_key", in_tensor[0].coordinate_map_key)
        setattr(m, "out_coordinate_map_key", out_coordinate_map_key)
        setattr(m, "coordinate_manager", in_tensor[0]._manager)

        return None

    @staticmethod
    def clean(m, force=False):
        if force:
            del m.in_shape, m.in_tensor, m.out_tensor, m.out_shape, \
                m.tensor_stride, m.in_coordinate_map_key, m.out_coordinate_map_key, \
                m.coordinate_manager

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
            out_feat, num_nonzero = fw_fn(
                input_features,
                m.pooling_mode,
                m.in_coordinate_map_key,
                m.out_coordinate_map_key,
                m.coordinate_manager._manager
            )

            setattr(m, "num_nonzero", num_nonzero)

        return out_feat.detach()

    @staticmethod
    def backward_pass(m, tensor, weight):
        with torch.no_grad():
            grad_out_feat = tensor.contiguous()

            bw_fn = get_minkowski_function('GlobalPoolingBackward', grad_out_feat)
            grad_in_feat = bw_fn(
                m.in_tensor,
                grad_out_feat,
                m.num_nonzero,
                m.pooling_mode,
                m.in_coordinate_map_key,
                m.out_coordinate_map_key,
                m.coords_manager._manager
            )

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

            fw_fn = get_minkowski_function('ConvolutionForward', input_features)
            out_feat = fw_fn(
                input_features,
                m.kernel,
                m.kernel_generator.kernel_size,
                m.kernel_generator.kernel_stride,
                m.kernel_generator.kernel_dilation,
                m.kernel_generator.region_type,
                m.kernel_generator.region_offsets,
                m.kernel_generator.expand_coordinates,
                m.convolution_mode,
                m.in_coordinate_map_key,
                m.out_coordinate_map_key,
                m.coordinate_manager._manager
            )

            out_feat.detach()

            return out_feat

    @staticmethod
    def backward_pass(m, in_tensor, weight, bias=None):
        grad_out_feat = in_tensor
        with torch.no_grad():
            if not grad_out_feat.is_contiguous():
                grad_out_feat = grad_out_feat.contiguous()

            bw_fn = get_minkowski_function('ConvolutionBackward', grad_out_feat)
            grad_in_feat, grad_kernel = bw_fn(
                m.in_tensor,
                grad_out_feat,
                m.kernel,
                m.kernel_generator.kernel_size,
                m.kernel_generator.kernel_stride,
                m.kernel_generator.kernel_dilation,
                m.kernel_generator.region_type,
                m.kernel_generator.region_offsets,
                m.convolution_mode,
                m.in_coordinate_map_key,
                m.out_coordinate_map_key,
                m.coordinate_manager._manager
            )

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
        input_features = in_tensor

        if not input_features.is_contiguous():
            input_features = input_features.contiguous()

        fw_fn = get_minkowski_function('ConvolutionTransposeForward', input_features)
        out_feat = fw_fn(
            input_features,
            m.kernel,
            m.kernel_generator.kernel_size,
            m.kernel_generator.kernel_stride,
            m.kernel_generator.kernel_dilation,
            m.kernel_generator.region_type,
            m.kernel_generator.region_offsets,
            m.kernel_generator.expand_coordinates,
            m.convolution_mode,
            m.in_coordinate_map_key,
            m.out_coordinate_map_key,
            m.coordinate_manager._manager
        )

        out_feat.detach()

        return out_feat

    @staticmethod
    def backward_pass(m, in_tensor, weight, bias=None):
        grad_out_feat = in_tensor
        with torch.no_grad():
            if not grad_out_feat.is_contiguous():
                grad_out_feat = grad_out_feat.contiguous()

            bw_fn = get_minkowski_function('ConvolutionTransposeBackward', grad_out_feat)
            grad_in_feat, grad_kernel = bw_fn(
                m.in_tensor,
                grad_out_feat,
                m.kernel,
                m.kernel_generator.kernel_size,
                m.kernel_generator.kernel_stride,
                m.kernel_generator.kernel_dilation,
                m.kernel_generator.region_type,
                m.kernel_generator.region_offsets,
                m.convolution_mode,
                m.in_coordinate_map_key,
                m.out_coordinate_map_key,
                m.coordinate_manager._manager
            )

            grad_in_feat.detach()

            assert grad_in_feat.size() == m.in_shape, (grad_in_feat.size(), m.in_shape(), m.out_shape, m.in_coords.size())

            return grad_in_feat
