#  NLP written by GAMS Convert at 04/21/18 13:54:23
#  
#  Equation counts
#      Total        E        G        L        N        X        C        B
#          7        4        0        3        0        0        0        0
#  
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#          7        7        0        0        0        0        0        0
#  FX      0        0        0        0        0        0        0        0
#  
#  Nonzero counts
#      Total    const       NL      DLL
#         21       18        3        0
# 
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *

model = m = ConcreteModel()


m.x1 = Var(within=Reals,bounds=(0,3),initialize=0)
m.x2 = Var(within=Reals,bounds=(0,4),initialize=0)
m.x3 = Var(within=Reals,bounds=(0,4),initialize=0)
m.x4 = Var(within=Reals,bounds=(0,2),initialize=0)
m.x5 = Var(within=Reals,bounds=(0,2),initialize=0)
m.x6 = Var(within=Reals,bounds=(0,6),initialize=0)

m.obj = Objective(expr=m.x1**0.6 + m.x2**0.6 + m.x3**0.4 - 4*m.x3 + 2*m.x4 + 5*m.x5 - m.x6, sense=minimize)

m.c1 = Constraint(expr= - 3*m.x1 + m.x2 - 3*m.x4 == 0)

m.c2 = Constraint(expr= - 2*m.x2 + m.x3 - 2*m.x5 == 0)

m.c3 = Constraint(expr=   4*m.x4 - m.x6 == 0)

m.c4 = Constraint(expr=   m.x1 + 2*m.x4 <= 4)

m.c5 = Constraint(expr=   m.x2 + m.x5 <= 4)

m.c6 = Constraint(expr=   m.x3 + m.x6 <= 6)
