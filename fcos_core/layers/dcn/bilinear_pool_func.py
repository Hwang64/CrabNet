import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from fcos_core import _C


class BilinearPoolingFunction(Function):

    @staticmethod
    def forward(
        ctx,
        data,
        offset,
        dimension,
        channel,
        height,
        width,
        trans_std=.0
    ):
        ctx.trans_std = trans_std

        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError

        output = data.new_empty(dimension, channel, height, width)
        _C.bilinear_pooling_forward(
            data,
            offset,
            output,
            ctx.trans_std
        )

        if data.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, offset)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        data, offset = ctx.saved_tensors
        grad_input = torch.zeros_like(data)
        grad_offset = torch.zeros_like(offset)
        _C.bilinear_pooling_backward(
            grad_output,
            data,
            offset,
            grad_input,
            grad_offset,
            ctx.trans_std
        )
        #print('data')
        #print(data)
        #print('offset')
        #print(offset)
        #print('grad_data')
        #print(grad_input)
        #print('grad_offset')
        #print(grad_offset)
        #print(grad_output.is_contiguous())
        return (grad_input, grad_offset, None, None, None, None, None)

bilinear_pooling = BilinearPoolingFunction.apply
