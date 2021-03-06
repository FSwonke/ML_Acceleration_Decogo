#  NLP written by GAMS Convert at 04/21/18 13:51:43
#  
#  Equation counts
#      Total        E        G        L        N        X        C        B
#         10        2        0        8        0        0        0        0
#  
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#          7        7        0        0        0        0        0        0
#  FX      0        0        0        0        0        0        0        0
#  
#  Nonzero counts
#      Total    const       NL      DLL
#         54       14       40        0
# 
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *

model = m = ConcreteModel()


m.x1 = Var(within=Reals,bounds=(1E-6,1),initialize=0.295)
m.x2 = Var(within=Reals,bounds=(1E-6,1),initialize=0.148)
m.x3 = Var(within=Reals,bounds=(1E-6,1),initialize=0.463)
m.x4 = Var(within=Reals,bounds=(1E-6,1),initialize=0.094)
m.x5 = Var(within=Reals,bounds=(20,80),initialize=57.154)
m.x7 = Var(within=Reals,bounds=(0,None),initialize=0)

m.obj = Objective(expr=   m.x7, sense=minimize)

m.c2 = Constraint(expr=log(m.x1 + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*m.x4) + m.x1/(m.x1
                        + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*m.x4) + 1.55190688128384*
                       m.x2/(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 1.27289874839144*m.x4) + 
                       0.767395887387844*m.x3/(0.767395887387844*m.x1 + 0.176307940228365*m.x2 + m.x3 + 
                       0.187999658986436*m.x4) + 0.989870205661735*m.x4/(0.989870205661735*m.x1 + 0.928335072476283*m.x2
                        + 0.308103094315467*m.x3 + m.x4) + 2787.49800065313/(229.664 + m.x5) - m.x7 <= 10.7545020354713)

m.c3 = Constraint(expr=log(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 1.27289874839144*m.x4) + 
                       1.2689544013438*m.x1/(m.x1 + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*
                       m.x4) + m.x2/(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 1.27289874839144*m.x4) + 
                       0.176307940228365*m.x3/(0.767395887387844*m.x1 + 0.176307940228365*m.x2 + m.x3 + 
                       0.187999658986436*m.x4) + 0.928335072476283*m.x4/(0.989870205661735*m.x1 + 0.928335072476283*m.x2
                        + 0.308103094315467*m.x3 + m.x4) + 2696.24885600287/(226.232 + m.x5) - m.x7 <= 10.3803549837107)

m.c4 = Constraint(expr=log(0.767395887387844*m.x1 + 0.176307940228365*m.x2 + m.x3 + 0.187999658986436*m.x4) + 
                       0.696334182309743*m.x1/(m.x1 + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*
                       m.x4) + 0.696676834276998*m.x2/(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 
                       1.27289874839144*m.x4) + m.x3/(0.767395887387844*m.x1 + 0.176307940228365*m.x2 + m.x3 + 
                       0.187999658986436*m.x4) + 0.308103094315467*m.x4/(0.989870205661735*m.x1 + 0.928335072476283*m.x2
                        + 0.308103094315467*m.x3 + m.x4) + 3643.31361767678/(239.726 + m.x5) - m.x7 <= 12.9738026256517)

m.c5 = Constraint(expr=log(0.989870205661735*m.x1 + 0.928335072476283*m.x2 + 0.308103094315467*m.x3 + m.x4) + 
                       0.590071729272002*m.x1/(m.x1 + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*
                       m.x4) + 1.27289874839144*m.x2/(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 
                       1.27289874839144*m.x4) + 0.187999658986436*m.x3/(0.767395887387844*m.x1 + 0.176307940228365*m.x2
                        + m.x3 + 0.187999658986436*m.x4) + m.x4/(0.989870205661735*m.x1 + 0.928335072476283*m.x2 + 
                       0.308103094315467*m.x3 + m.x4) + 2755.64173589155/(219.161 + m.x5) - m.x7 <= 10.2081676704566)

m.c6 = Constraint(expr=(-log(m.x1 + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*m.x4)) - (m.x1/(
                       m.x1 + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*m.x4) + 1.55190688128384
                       *m.x2/(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 1.27289874839144*m.x4) + 
                       0.767395887387844*m.x3/(0.767395887387844*m.x1 + 0.176307940228365*m.x2 + m.x3 + 
                       0.187999658986436*m.x4) + 0.989870205661735*m.x4/(0.989870205661735*m.x1 + 0.928335072476283*m.x2
                        + 0.308103094315467*m.x3 + m.x4)) - 2787.49800065313/(229.664 + m.x5) - m.x7
                        <= -10.7545020354713)

m.c7 = Constraint(expr=(-log(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 1.27289874839144*m.x4)) - (
                       1.2689544013438*m.x1/(m.x1 + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*
                       m.x4) + m.x2/(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 1.27289874839144*m.x4) + 
                       0.176307940228365*m.x3/(0.767395887387844*m.x1 + 0.176307940228365*m.x2 + m.x3 + 
                       0.187999658986436*m.x4) + 0.928335072476283*m.x4/(0.989870205661735*m.x1 + 0.928335072476283*m.x2
                        + 0.308103094315467*m.x3 + m.x4)) - 2696.24885600287/(226.232 + m.x5) - m.x7
                        <= -10.3803549837107)

m.c8 = Constraint(expr=(-log(0.767395887387844*m.x1 + 0.176307940228365*m.x2 + m.x3 + 0.187999658986436*m.x4)) - (
                       0.696334182309743*m.x1/(m.x1 + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*
                       m.x4) + 0.696676834276998*m.x2/(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 
                       1.27289874839144*m.x4) + m.x3/(0.767395887387844*m.x1 + 0.176307940228365*m.x2 + m.x3 + 
                       0.187999658986436*m.x4) + 0.308103094315467*m.x4/(0.989870205661735*m.x1 + 0.928335072476283*m.x2
                        + 0.308103094315467*m.x3 + m.x4)) - 3643.31361767678/(239.726 + m.x5) - m.x7
                        <= -12.9738026256517)

m.c9 = Constraint(expr=(-log(0.989870205661735*m.x1 + 0.928335072476283*m.x2 + 0.308103094315467*m.x3 + m.x4)) - (
                       0.590071729272002*m.x1/(m.x1 + 1.2689544013438*m.x2 + 0.696334182309743*m.x3 + 0.590071729272002*
                       m.x4) + 1.27289874839144*m.x2/(1.55190688128384*m.x1 + m.x2 + 0.696676834276998*m.x3 + 
                       1.27289874839144*m.x4) + 0.187999658986436*m.x3/(0.767395887387844*m.x1 + 0.176307940228365*m.x2
                        + m.x3 + 0.187999658986436*m.x4) + m.x4/(0.989870205661735*m.x1 + 0.928335072476283*m.x2 + 
                       0.308103094315467*m.x3 + m.x4)) - 2755.64173589155/(219.161 + m.x5) - m.x7 <= -10.2081676704566)

m.c10 = Constraint(expr=   m.x1 + m.x2 + m.x3 + m.x4 == 1)
