POD-PINN
=====
# Method abstract
A physics-informed machine learning framework is developed for the reduced-order modeling of
parametrized steady-state partial differential equations (PDEs). During the offline stage, a reduced
basis is extracted from a collection of high-fidelity solutions and a reduced-order model is then
constructed by a Galerkin projection of the full-order model onto the reduced space. A feedforward
neural network is used to approximate the mapping from the physical/geometrical parameters to
the reduced coefficients. The network can be trained by minimizing the mean squared residual error
of the reduced-order equation on a set of points in parameter space. Such a network is referred to
as physics-informed neural network (PINN). As the number of residual points is unlimited, a large
data set can be generated to train a PINN to approximate the reduced-order model. However, the
accuracy of such a network is often limited. This is improved by using the high-fidelity solutions
that are generated to extract the reduced basis. The network is then trained by minimizing the
sum of the mean squared residual error of the reduced-order equation and the mean squared error
between the network output and the projection coefficients of the high-fidelity solutions. For complex
nonlinear problems, the projection of high-fidelity solution onto the reduced space is more accurate
than the solution of the reduced-order equation. Therefore, higher accuracy than the PINN for this
network - referred to as physics-reinforced neural network (PRNN) - can be expected for complex
nonlinear problems. Numerical results demonstrate that the PRNN is more accurate than the PINN
and both are more accurate than a purely data-driven neural network (referred to PDNN) for complex problems. During
the reduced basis refinement, before reaching its accuracy limit, the PRNN obtains higher accuracy
than the direct reduced-order model based on a Galerkin projection.

The relationship of these methods is dipicted in the follwing.

<p align="center">
  <img src="https://github.com/cwq2016/POD-PINN/blob/master/IMG/RelationshipChart.jpg" height="500px">
</p>

# Code instruction
The code is written in Python, and the feedforward neural networks are built based on PyTorch framework. The code tries to implement the PDNN, PINN and PRNN for three test cases:
* The one-dimensional Burgers’ equation
* The two-dimensional lid-driven cavity flow
* The two-dimensional natural convection
The high-fidelity solver(Chebyshev pseudospectral solver) is out of the scope of this code. In each test case, the high-fidelity solutions are precomputed with HF solver and stored in the "Numsols" under the folder for each case. The user can use his/her own HF solver instead. But the reduced-order model built in "Net*.py"(* denote case name), should be revised accordingly.

The code can run both on CPU and single GPU. The user can run "Cases_test.py" to do test serially, or utilize the "batch.sh" for parallel test. After training the neural networks, the "accuracy_comparsion.py" to gain an view of the error of different networks.

# Reference
This is a scientific project. If you use POD-PINN for publications or presentations in science, please support the project by citing our publications given in references. In case you have question regarding POD-PINN, don't hesitate to contact us wenqianchen2016@gmail.com.
