import torch
from torch import tensor

class NDVILayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: tensor) -> tensor:
        """
        input = [[a,b]]
        ndvi= (b-a)/(b+a)+1e-8
        """
        ctx.save_for_backward(input, None)
        addMatrix = tensor([[1.0, 1.0]], requires_grad=True).t()
        minusMatrix = tensor([[-1.0, 1.0]], requires_grad=True).t()
        sum = torch.matmul(input, addMatrix)
        diff = torch.matmul(input, minusMatrix)
        ndvi = diff / sum + 1e-8
        output = tensor([[ndvi]], requires_grad=True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        partial_a = -2b/(a+b)^2
        partial_b = 2a/(a+b)^2
        """
        input, _ = ctx.saved_tensors
        base = torch.sum(input, dim=1)
        base = base * base
        mul_matrix = torch.tensor([[0.0, -1.0],
                                   [1.0, 0.0]])
        upper = torch.matmul(mul_matrix, input.t())
        result = 2 * upper / base
        return grad_output * result.t()


class NDVILoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, labels):
        assert input.shape == (1, 1)
        ctx.save_for_backward(input, labels)
        if labels[0] == 1:
            if 0.3 < input[0][0] < 0.8:
                return tensor([0]).float()
            elif input[0][0] <= 0.3:
                return tensor([0.8 - input[0][0]]).float()
            else:
                return tensor([input[0][0] - 0.3]).float()
        elif labels[0] == 0:
            return tensor([max(abs(input[0][0] - 1), abs(input[0][0] - (-1)))]).float()

    @staticmethod
    def backward(ctx, output_grad):
        input, labels = ctx.saved_tensors
        assert input.shape == (1, 1)
        result = input.clone().detach()
        if labels[0] == 1:
            if 0.3 < input[0][0] < 0.8:
                result[0][0] = 1e-8
            elif input[0][0] <= 0.3:
                result[0][0] = -1.0
            else:
                result[0][0] = 1.0
        elif labels[0] == 0:
            if input[0][0] < 0:
                result[0][0] = -1.0
            else:
                result[0][0] = 1.0
        return output_grad * result.float(), None
