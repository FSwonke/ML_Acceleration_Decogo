#  NLP written by GAMS Convert at 04/21/18 13:51:44
#  
#  Equation counts
#      Total        E        G        L        N        X        C        B
#          7        5        0        2        0        0        0        0
#  
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#         10       10        0        0        0        0        0        0
#  FX      0        0        0        0        0        0        0        0
#  
#  Nonzero counts
#      Total    const       NL      DLL
#         30       23        7        0
# 
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *

model = m = ConcreteModel()


m.x1 = Var(within=Reals,bounds=(0,100),initialize=0)
m.x2 = Var(within=Reals,bounds=(0,200),initialize=0)
m.x3 = Var(within=Reals,bounds=(0,500),initialize=0)
m.x4 = Var(within=Reals,bounds=(0,500),initialize=0)
m.x5 = Var(within=Reals,bounds=(0,500),initialize=0)
m.x6 = Var(within=Reals,bounds=(0,500),initialize=0)
m.x7 = Var(within=Reals,bounds=(0,500),initialize=0)
m.x8 = Var(within=Reals,bounds=(0,500),initialize=0)
m.x9 = Var(within=Reals,bounds=(0,500),initialize=0)

m.obj = Objective(expr= - 9*m.x1 - 15*m.x2 + 6*m.x3 + 13*m.x4 + 10*m.x5 + 10*m.x6, sense=minimize)

m.c2 = Constraint(expr= - m.x3 - m.x4 + m.x8 + m.x9 == 0)

m.c3 = Constraint(expr=   m.x1 - m.x5 - m.x8 == 0)

m.c4 = Constraint(expr=   m.x2 - m.x6 - m.x9 == 0)

m.c5 = Constraint(expr=m.x7*m.x8 - 2.5*m.x1 + 2*m.x5 <= 0)

m.c6 = Constraint(expr=m.x7*m.x9 - 1.5*m.x2 + 2*m.x6 <= 0)

m.c7 = Constraint(expr=m.x7*m.x8 + m.x7*m.x9 - 3*m.x3 - m.x4 == 0)
