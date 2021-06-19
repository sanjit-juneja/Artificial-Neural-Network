from enum import Enum
import numpy as np
import warnings
"""
This file manages the various activation functions that we might use for the neurons in our ANN. We also manage
the derivative functions for these activations functions. For the ones we are using, it turns out that it is
easier to define f'(x) as a function of f(x). That is f'(x) = g(f(x))... and we are defining g().

Each layer might have a different activation function. When creating an ANN, we will have to provide a tuple of 
Activation_Function constants to refer to these functions, one constant per layer. In the ANN, itself, it will 
create two lists, a list of activation functions and a list of derivative functions. This is a little weird - 
you probably haven't met arrays of functions before! (But in Python, you can do that.)
"""




# The Activation_Function enumeration is kind of a fancy way of making sequential constants that we can
# share between files.
class Activation_Type(Enum):
    IDENTITY = 0
    SIGMOID = 1
    TANH = 2
    RELU = 3
    LEAKY_RELU = 4
    ELU = 5

"""
The identity activation function takes whatever the total input was to the neuron and sends that directly to
the output (axons). Since it is essentially f(x) = x, the derivative is g(f(x)) = 1.
Identity has the advantage of having an unconstrained domaian and range.
"""
def identity(a):
    return a

def delta_identity_from_identity(y):
    return np.ones(y.shape)

"""
Sigmoid is one of the most famous activation functions- the asymtote for x = -infinity is 0 and 
for x = +infinity, it is 1, with a smooth transition around x = 0. It's derivative is remarkable - 
it turns out that g(f(x)) = (1-f(x))*(f(x))
The signmoid's output is constrained to values between 0 and 1.
"""
def sigmoid(a):
    return 1/(1+np.exp(-1*a))

def delta_sigmoid_from_sigmoid(y):
    return (1-y)*y
"""
The hyperbolic tangent is similar to the sigmoid in shape, but while its slope gets flatter 
as |x| gets large, it doesn't have an asymptope. 
The tanh activation function isn't quite constrained to within 0 and 1, but values don't stay far from it.
"""
def tanh(a):
    return np.tanh(a)

def delta_tanh_from_tanh(y):
    return 1 - np.power(y,2)

"""
The Relu Activation function is the identity activation function for x>0 and 0 for x<0.
This gives it zero slope when x<0 and slope = 1 when x>0. Its output is constrained to be non-negative.
"""
def relu(a):
    return np.maximum(a,0)

def delta_relu_from_relu(y):
    return np.zeros(y.shape,dtype=int)+np.ones(y.shape,dtype=int)*(y>0)

"""
The Leaky Relu Activation function is similar to the Relu, as its name would suggest. However, when
x is negative, it has a result f(x) = beta * x, where beta is typically a very small number, so the
negative side of the graph is a shallow, straight line. If this beta was zero, the leaky Relu becomes
the plain Relu. If beta is 1, then leaky Relu becomes the identity.
The output of leaky relu is not constrained.
"""
def leaky_relu(a, leaky_factor = 0.01):

    return np.maximum(a,a*leaky_factor)

def delta_leaky_relu_from_relu(y, leaky_factor = 0.01):
     return np.ones(y.shape,dtype=int)*(y>0)+np.ones(y.shape,dtype=int)*leaky_factor*(y<=0)

"""
The Exponential Linear Unit is also similar to LeakyRELU, but with a smoother curve and a asymptotic negative value.
If a>0, ELU -> a; if a<=0, ELU -> e^a - 1
The output of elu is (-alpha, infinity)
"""
def elu(a, alpha = 1):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            z = np.minimum(a,0)
            result = a*(a>0)+alpha*(np.exp(z)-1)*(a<=0)
        except Warning as e:
            print('error found:', e)
    return result

def delta_elu(y,alpha =1):
    return np.minimum(np.ones(y.shape),y+alpha)


"""
These are dictionaries of the enumerated constants to the functions we just defined, so we can access
the functions programmatically from the constants without a lot of if statements. (Or switch-case, which 
Python doesn't have. Again, this is a new way of thinking about functions - in this case the values in a 
dictionary!
So these dictionaries are [Activation_Function -> function]
"""


ACTIVATION= {Activation_Type.IDENTITY: identity,
             Activation_Type.SIGMOID: sigmoid,
             Activation_Type.TANH: tanh,
             Activation_Type.RELU: relu,
             Activation_Type.LEAKY_RELU: leaky_relu,
             Activation_Type.ELU: elu}

DERIVATIVE_FROM_ACTIVATION = {Activation_Type.IDENTITY: delta_identity_from_identity,
                              Activation_Type.SIGMOID: delta_sigmoid_from_sigmoid,
                              Activation_Type.TANH: delta_tanh_from_tanh,
                              Activation_Type.RELU: delta_relu_from_relu,
                              Activation_Type.LEAKY_RELU: delta_leaky_relu_from_relu,
                              Activation_Type.ELU: delta_elu}