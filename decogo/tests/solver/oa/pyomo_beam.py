from pyomo.environ import *  # import pyomo module
import math  # import math module
from decogo.solver.decogo import DecogoSolver

# calculate stress s1 in beam 1 and 3
def s1(a1, a2):
    return (P * cos(math.radians(theta))) / (sqrt(2) * a1) + (P * sin(math.radians(theta))) / (
                sqrt(2) * (a1 + sqrt(2) * a2))


# calculate stress s2 in beam 2
def s2(a1, a2):
    return (sqrt(2) * P * sin(math.radians(theta))) / (a1 + sqrt(2) * a2)


# define predefined values
P = 50000  # [N] load
theta = 30  # [°] angle of load
rho = 7800  # [kg/m³] density
sig_max = 150 #* (10 ** 6)  # [Pa]=[N/m²] maximum stress for a beam
L = 1  # [m] length of the beam

model = ConcreteModel()  # create instance of model

# create first independent variable named A1 which defines the cross-sectional area of beam 1 and 3
# the variable has the lower bound 0 but has no upper bound
# the value of the variable has to be positive
model.A1 = Var(within=NonNegativeReals, 
               bounds=(0, None))

# create first independent variable named A2 which defines the cross-sectional area of beam 2
# the variable has the lower bound 0 but has no upper bound
# the value of the variable has to be positive
model.A2 = Var(within=NonNegativeReals,
               bounds=(0, None))

# define the objective
# objective: minimize mass function which depends on beams cross-sectional areas A1 and A2
model.m = Objective(
    expr=rho * L * (sqrt(2) *2* model.A1 + model.A2),
    sense=minimize
)

# define constraints
# constraints are given through actual stress in beams which can not be higher than sigma_max
# constraint for beam 1 and 3
model.c1 = Constraint(
    expr=(0, s1(model.A1, model.A2), sig_max)
)

# constraint for beam 2
model.c2 = Constraint(
    expr=(0, s2(model.A1, model.A2), sig_max)
)

opt = SolverFactory('decogo')  # create the solver
result_obj = opt.solve(model, tee=True)  # solve the problem

model.display()  # print the results
