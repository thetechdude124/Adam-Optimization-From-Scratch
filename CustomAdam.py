import torch
from torch.optim import Optimizer

#Custom Adam Optimizer - extension of Optimizer class
class CustomAdam(Optimizer):
    
    """
    A custom implementation of the Adam optimizer. Defaults used are as recommended in https://arxiv.org/abs/1412.6980 
    See the paper or visit Optimizer_Experimentation.ipynb for more information on how exactly Adam works + mathematics behind it.

    Params:
    stepsize (float): the effective upperbound of the optimizer step in most cases (size of step). DEFAULT - 0.001.
    bias_m1 (float): bias for the first moment estimate. DEFAULT - 0.9
    bias_m2 (float): bias for the second uncentered moment estimate, DEFAULT - 0.999.
    epsilon (float): small number added to prevent division by zero, DEFAULT - 10e-8.
    bias_correction (bool): whether the optimizer should correct for the specified biases when taking a step. DEFAULT - TRUE.
    """
    #Initialize optimizer with parameters
    def __init__(self, params, stepsize = 0.001, bias_m1 = 0.9, bias_m2 = 0.999, epsilon = 10e-8, bias_correction = True):
        #Check if stepsizes and biases are invalid (negative)
        if stepsize < 0:
            raise ValueError("Invalid stepsize [{}]. Choose a positive stepsize".format(stepsize))
        if bias_m1 < 0 or bias_m2 < 0 and bias_correction:
            raise ValueError("Invalid bias parameters [{}, {}]. Choose positive bias parameters.".format(bias_m1, bias_m2))
        #Declare dictionary of default values for optimizer initialization
        DEFAULTS = dict(stepsize = stepsize, bias_m1 = bias_m1, bias_m2 = bias_m2, epsilon = epsilon, bias_correction = bias_correction)
        #Initialize the optimizer
        super(CustomAdam, self).__init__(params, DEFAULTS)

    #Step method (for updating parameters)
    def step(self, closure = None):
        #Set loss to none
        loss = None
        #If the closure is set to True, set the loss to the closure function
        loss = closure() if closure == None else loss
        #Check if this is the first step - if not, increment the current step
        if self.state["step"] == None:
            self.state["step"] = 1
        else:
            self.state["step"] += 1
        #Iterate over "groups" of parameters (layers of parameters in the network) to begin processing and computing the next set of params
        for param_group in self.param_groups:
            #Check if gradients have been computed for each group of parameters
            #If not, skip over that group
            if param_group["params"].grad.data == None:
                continue
            else: 
                gradients = param_group["params"].grad.data
            #Use Adam optimization method - first, define all the required arguments for the parameter if we are on the first step
            if self.state["step"] == 1:
                #Set the first and second moment estimates to zeroes
                self.state["first_moment_estimate"] = torch.zeros_like(param_group.data)
                self.state["second_moment_estimate"] = torch.zeros_like(param_group.data)
            #Declare variables from state - inplace methods modify state variable directly
            first_moment_estimate = self.state["first_moment_estimate"]
            second_moment_estimate = self.state["second_moment_estimate"]
            #Compute the first moment estimate - B_1 * m_t + (1-B_1) * grad (uncentered)
            first_moment_estimate.matmul_(param_group["bias_m1"]).addmm_(gradients, 1.0 - param_group["bias_m1"])
            #Compute the second moment estimate - B_2 * v_t + (1-B_2) * grad^2 (uncentered)
            second_moment_estimate.matmul_(param_group["bias_m2"]).addmm_(gradients.float_power_(2.0), 1.0 - param_group["bias_m2"])
            #Perform bias correction if parameter is set to true
            if param_group["bias_correction"]:
                #Perform bias correction for the first moment estimate: m_t / (1 -B_1^t)
                first_moment_estimate.divide_(1.0 - (param_group["bias_m1"] ** self.state["step"]))
                #Perform bias correction for second moment estimate: v_t / (1 - B_2^t)
                second_moment_estimate.divide_(1.0 - (param_group["bias_m2"] ** self.state["step"]))
            #Next, perform the actual update
            #Multiply the stepsize a by the quotient of the first moment estimate and the square root of the second moment estimate plus epsilon
            #In other words - theta = theta_{t-1} - a * first_estimate/(sqr(second_estimate) + epsilon)
            param_group.data.add_((-param_group["stepsize"]) * first_moment_estimate.divide_(second_moment_estimate.sqrt_() + param_group["epsilon"]))
        #Return the loss
        return loss