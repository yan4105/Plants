from CustomLayers import NDVILoss, NDVILayer
import torch
import random

class MyContext:
    def __init__(self):
        self.args = None

    def save_for_backward(self, *args):
        self.args = args

    @property
    def saved_tensors(self):
        result = self.args
        self.args = None
        return result

def test_ndviloss():
    def create_inputs(num):
        rands = []
        inputs = []
        for i in range(num):
            rands.append(random.uniform(-1,1))
        for num in rands:
            inputs.append(torch.tensor([[num]]))
        return inputs
    inputs = create_inputs(100)
    labels = [[1],[0]]
    layer = NDVILoss
    delta = 1e-4
    check_delta = 1
    def error_msg(error, input, label, gradient, backward_gradient):
        return "error is: {error}, input: {input}, label: {label}, gradient: {gradient} \"" \
               "backward_gradient: {backward_gradient}".format(error=error, input=input, label=label, gradient=gradient,\
                                                               backward_gradient=backward_gradient)
    for input in inputs:
        for label in labels:
            input_delta = input + delta
            ctx = MyContext()
            output = layer.forward(ctx, input, label)
            backward_gradient, _ = layer.backward(ctx, 1)
            output_delta = layer.forward(ctx, input_delta, label)
            gradient = (output_delta - output) / delta
            error = abs(gradient-backward_gradient)
            assert error <= check_delta, error_msg(error, input, label, gradient, backward_gradient)

def test_ndvilayer():
    def get_inputs(num):
        r = 50
        inputs = []
        for _ in range(num):
            a,b = random.uniform(-r, r), random.uniform(-r, r)
            inputs.append(torch.tensor([[a,b]]))
        return inputs
    inputs = get_inputs(100)
    layer = NDVILayer
    delta = 1e-5
    check_delta = 10
    def error_msg(error, gradient, backward_gradient, input):
        return "input: {input}, error is: {error}, gradient is: {gradient}, backward_gradient is: {backward_gradient}".\
            format(input=input, error=error, gradient=gradient, backward_gradient=backward_gradient)
    for input in inputs:
        gradient = input.clone()
        ctx = MyContext()
        output = layer.forward(ctx, input)
        backward_gradient = layer.backward(ctx, 1)
        for i in range(input.shape[1]):
            print("input:", input)
            input_delta = input.clone()
            input_delta[0][i] += delta
            print("input after:", input)
            print("input_delta:", input_delta)
            output_delta = layer.forward(ctx, input_delta)
            gradient_i = (output_delta - output) / delta
            print("gradient_i:", gradient_i)
            gradient[0][i] = gradient_i[0][0]
        error = abs(gradient-backward_gradient)
        assert error[0][0] <= check_delta and error[0][1] <= check_delta, \
                error_msg(error,gradient,backward_gradient, input)

if __name__ == '__main__':
    test_ndviloss()
    test_ndvilayer()