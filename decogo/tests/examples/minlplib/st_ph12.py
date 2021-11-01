#  NLP written by GAMS Convert at 04/21/18 13:54:24
#  
#  Equation counts
#      Total        E        G        L        N        X        C        B
#          5        1        0        4        0        0        0        0
#  
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#          4        4        0        0        0        0        0        0
#  FX      0        0        0        0        0        0        0        0
#  
#  Nonzero counts
#      Total    const       NL      DLL
#         10        7        3        0
# 
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *

model = m = ConcreteModel()


m.x1 = Var(within=Reals,bounds=(0,None),initialize=0)
m.x2 = Var(within=Reals,bounds=(0,None),initialize=0)
m.x3 = Var(within=Reals,bounds=(0,None),initialize=0)

m.obj = Objective(expr=m.x1 - 0.5*m.x1**2 - m.x2**2 + 2*m.x2 - 1.5*m.x3**2 + 3*m.x3, sense=minimize)

m.c1 = Constraint(expr=   m.x1 <= 4)

m.c2 = Constraint(expr=   m.x2 <= 4)

m.c3 = Constraint(expr=   m.x3 <= 4)

m.c4 = Constraint(expr=   2*m.x1 + 3*m.x2 + 4*m.x3 <= 35)
