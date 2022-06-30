### **📉ADAM OPTIMIZATION FROM SCRATCH.**

*Purpose: Implementing the ADAM optimizer from the ground up with PyTorch and comparing its performance on 6 3-D objective functions (each progressively more difficult to optimize) against SGD, AdaGrad, and RMSProp.*

In recent years, the Adam optimizer has become famous for achieving fast and accurate results when it comes to optimizing complex stochastic loss functions - thanks to its moment estimates (as I'll explain further) and update rule, it is able to more efficiently converge to reliable local (and sometimes global) minima, and has been shown to perform remarkably well on high-dimensional objective functions due to **a very small memory requirement - the optimization method is invariant to gradient scaling and does NOT need to compute higher order derivatives, thus making it more computationally efficient.**

The goal of this project is to 1️⃣ learn how optimization methods work mathematically and their theoretical behaviour (reading and taking notes on the paper + building a strong foundation of statistics), 2️⃣ apply this theoretical knowledge by constructing said optimizer from scratch in PyTorch (found in CustomAdam.py), and 3️⃣ test this custom implementation on six 3D functions against optimizers to determine performance. The objective of this last step was to see if I could **leverage the mathematical knowledge gained in a) to improve the optimizer. (Optimizer_Experimentation.ipynb).** 

This project was started in an effort to replicate the original 2017 paper! Check it out for more information and a more detailed explanation of how everything works under the hood - https://arxiv.org/pdf/1412.6980.pdf.

### ➗ How Adam Actually Works - The Math.

**Fundamentally, the core aspect of ML that allows machine to "learn" is optimization** - taking a differentiable objective (loss) function and attempting to intelligently modify the model's parameters in such a way that it results in a lower (or higher) value of this function. This is done iteratively - the optimization algorithm uses something known as the **gradient** (simply the derivative of the loss function at that point) to determine how best to update all the parameters. Each one of these updates/iterations is known as a **"step"**, and over a large number of steps, the model should ideally **converge to a local minima that offers arbitrarily good parameters.** The *size* of these steps is known as the *stepsize or learning rate*, as we'll go over next.

Here's a quick GIF using the standard SGD (Stochastic Gradient Descent) Optimization Algorithm (source: https://mlfromscratch.com/optimizers-explained/):

<p align = "center"><img src = "./images/GRADIENT-DESCENT-GIF.gif"></img></p>

**The key disadvantage of this method is that it becomes computationally expensive extremely fast.** The image above showed a model with one parameter being optimized - but, practically, most models have **dozens if not hundreds of thousands of parameters** that must be optimized. In such cases, we have to compute the **nth order derivative of some cost function $Z(θ)$** where $θ$ is a vector of all of these parameters - a task that must be repeated **for each step we take.** This quite evidently slows down the optimization process and drastically increases the compute needed as well.

How does Adam fix this? *By using first and second moment ESTIMATES to update parameters rather than the gradient themselves.* 

(**Quickly, here are some terms worth noting:**

- **Expected Value:** this is given by $E[X]=∫^∞_{-∞}x*f(x)dx$, and its fundamental goal is to represent **the return you can obtain for performing some action.** In this case, $f(x)$ is the probability distribution of some random variable, and *x* is the variable itself. All in all, this gives the probability that some random event will occur.
)