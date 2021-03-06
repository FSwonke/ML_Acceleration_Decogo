#  NLP written by GAMS Convert at 04/21/18 13:51:12
#  
#  Equation counts
#      Total        E        G        L        N        X        C        B
#         78       68       10        0        0        0        0        0
#  
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#         87       87        0        0        0        0        0        0
#  FX      0        0        0        0        0        0        0        0
#  
#  Nonzero counts
#      Total    const       NL      DLL
#        622      182      440        0
# 
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *

model = m = ConcreteModel()


m.x1 = Var(within=Reals,bounds=(0,None),initialize=0)
m.x2 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x3 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x4 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x5 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x6 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x7 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x8 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x9 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x10 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x11 = Var(within=Reals,bounds=(0,10000),initialize=1)
m.x12 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x13 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x14 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x15 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x16 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x17 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x18 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x19 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x20 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x21 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x22 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x23 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x24 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x25 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x26 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x27 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x28 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x29 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x30 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x31 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x32 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x33 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x34 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x35 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x36 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x37 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x38 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x39 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x40 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x41 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x42 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x43 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x44 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x45 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x46 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x47 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x48 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x49 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x50 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x51 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x52 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x53 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x54 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x55 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x56 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x57 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x58 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x59 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x60 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x61 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x62 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x63 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x64 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x65 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x66 = Var(within=Reals,bounds=(0,1),initialize=0.01)
m.x67 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x68 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x69 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x70 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x71 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x72 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x73 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x74 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x75 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x76 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x77 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x78 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x79 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x80 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x81 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x82 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x83 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x84 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x85 = Var(within=Reals,bounds=(0,1000),initialize=0)
m.x86 = Var(within=Reals,bounds=(0,1000),initialize=0)

m.obj = Objective(expr=   m.x67 + m.x68 + m.x69 + m.x70 + m.x71 + m.x72 + m.x73 + m.x74 + m.x75 + m.x76 + m.x77 + m.x78
                        + m.x79 + m.x80 + m.x81 + m.x82 + m.x83 + m.x84 + m.x85 + m.x86, sense=minimize)

m.c1 = Constraint(expr=0.5*m.x2 - m.x12*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 0.249999992497001*
                       m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*m.x8 + 
                       0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c2 = Constraint(expr=0.5000500050005*m.x3 - m.x13*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 
                       0.249999992497001*m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*
                       m.x8 + 0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c3 = Constraint(expr=0.24997499749975*m.x3 - m.x14*(0.24997499749975*m.x3 + 0.375037503750375*m.x4 + 0.375075018755251
                       *m.x5 + 0.312562515629377*m.x6 + 0.234398432806401*m.x7 + 0.164046067482601*m.x8 + 
                       0.109331212792365*m.x9 + 0.0702562148313726*m.x10 + 0.0438881594251137*m.x11) == 0)

m.c4 = Constraint(expr=0.375037503750375*m.x4 - m.x15*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 
                       0.249999992497001*m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*
                       m.x8 + 0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c5 = Constraint(expr=0.375037503750375*m.x4 - m.x16*(0.24997499749975*m.x3 + 0.375037503750375*m.x4 + 
                       0.375075018755251*m.x5 + 0.312562515629377*m.x6 + 0.234398432806401*m.x7 + 0.164046067482601*m.x8
                        + 0.109331212792365*m.x9 + 0.0702562148313726*m.x10 + 0.0438881594251137*m.x11) == 0)

m.c6 = Constraint(expr=0.124962496249625*m.x4 - m.x17*(0.124962496249625*m.x4 + 0.249999992497001*m.x5 + 
                       0.312562515629377*m.x6 + 0.312593787516885*m.x7 + 0.273519564077274*m.x8 + 0.218793754368866*m.x9
                        + 0.164062470437224*m.x10 + 0.117152290970943*m.x11) == 0)

m.c7 = Constraint(expr=0.249999992497001*m.x5 - m.x18*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 
                       0.249999992497001*m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*
                       m.x8 + 0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c8 = Constraint(expr=0.375075018755251*m.x5 - m.x19*(0.24997499749975*m.x3 + 0.375037503750375*m.x4 + 
                       0.375075018755251*m.x5 + 0.312562515629377*m.x6 + 0.234398432806401*m.x7 + 0.164046067482601*m.x8
                        + 0.109331212792365*m.x9 + 0.0702562148313726*m.x10 + 0.0438881594251137*m.x11) == 0)

m.c9 = Constraint(expr=0.249999992497001*m.x5 - m.x20*(0.124962496249625*m.x4 + 0.249999992497001*m.x5 + 
                       0.312562515629377*m.x6 + 0.312593787516885*m.x7 + 0.273519564077274*m.x8 + 0.218793754368866*m.x9
                        + 0.164062470437224*m.x10 + 0.117152290970943*m.x11) == 0)

m.c10 = Constraint(expr=0.0624624981253751*m.x5 - m.x21*(0.0624624981253751*m.x5 + 0.156218732806561*m.x6 + 
                        0.234398432806401*m.x7 + 0.273519564077274*m.x8 + 0.273546935193458*m.x9 + 0.246192241674115*
                        m.x10 + 0.205139666893903*m.x11) == 0)

m.c11 = Constraint(expr=0.156218732806561*m.x6 - m.x22*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 
                        0.249999992497001*m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*
                        m.x8 + 0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c12 = Constraint(expr=0.312562515629377*m.x6 - m.x23*(0.24997499749975*m.x3 + 0.375037503750375*m.x4 + 
                        0.375075018755251*m.x5 + 0.312562515629377*m.x6 + 0.234398432806401*m.x7 + 0.164046067482601*
                        m.x8 + 0.109331212792365*m.x9 + 0.0702562148313726*m.x10 + 0.0438881594251137*m.x11) == 0)

m.c13 = Constraint(expr=0.312562515629377*m.x6 - m.x24*(0.124962496249625*m.x4 + 0.249999992497001*m.x5 + 
                        0.312562515629377*m.x6 + 0.312593787516885*m.x7 + 0.273519564077274*m.x8 + 0.218793754368866*
                        m.x9 + 0.164062470437224*m.x10 + 0.117152290970943*m.x11) == 0)

m.c14 = Constraint(expr=0.156218732806561*m.x6 - m.x25*(0.0624624981253751*m.x5 + 0.156218732806561*m.x6 + 
                        0.234398432806401*m.x7 + 0.273519564077274*m.x8 + 0.273546935193458*m.x9 + 0.246192241674115*
                        m.x10 + 0.205139666893903*m.x11) == 0)

m.c15 = Constraint(expr=0.0312187515640633*m.x6 - m.x26*(0.0312187515640633*m.x6 + 0.0937031062453125*m.x7 + 
                        0.164046067482601*m.x8 + 0.218793754368866*m.x9 + 0.246192241674115*m.x10 + 0.246216883075546*
                        m.x11) == 0)

m.c16 = Constraint(expr=0.0937031062453125*m.x7 - m.x27*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 
                        0.249999992497001*m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*
                        m.x8 + 0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c17 = Constraint(expr=0.234398432806401*m.x7 - m.x28*(0.24997499749975*m.x3 + 0.375037503750375*m.x4 + 
                        0.375075018755251*m.x5 + 0.312562515629377*m.x6 + 0.234398432806401*m.x7 + 0.164046067482601*
                        m.x8 + 0.109331212792365*m.x9 + 0.0702562148313726*m.x10 + 0.0438881594251137*m.x11) == 0)

m.c18 = Constraint(expr=0.312593787516885*m.x7 - m.x29*(0.124962496249625*m.x4 + 0.249999992497001*m.x5 + 
                        0.312562515629377*m.x6 + 0.312593787516885*m.x7 + 0.273519564077274*m.x8 + 0.218793754368866*
                        m.x9 + 0.164062470437224*m.x10 + 0.117152290970943*m.x11) == 0)

m.c19 = Constraint(expr=0.234398432806401*m.x7 - m.x30*(0.0624624981253751*m.x5 + 0.156218732806561*m.x6 + 
                        0.234398432806401*m.x7 + 0.273519564077274*m.x8 + 0.273546935193458*m.x9 + 0.246192241674115*
                        m.x10 + 0.205139666893903*m.x11) == 0)

m.c20 = Constraint(expr=0.0937031062453125*m.x7 - m.x31*(0.0312187515640633*m.x6 + 0.0937031062453125*m.x7 + 
                        0.164046067482601*m.x8 + 0.218793754368866*m.x9 + 0.246192241674115*m.x10 + 0.246216883075546*
                        m.x11) == 0)

m.c21 = Constraint(expr=0.0156015671898445*m.x7 - m.x32*(0.0156015671898445*m.x7 + 0.0546382681253312*m.x8 + 
                        0.109331212792365*m.x9 + 0.164062470437224*m.x10 + 0.205139666893903*m.x11) == 0)

m.c22 = Constraint(expr=0.0546382681253312*m.x8 - m.x33*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 
                        0.249999992497001*m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*
                        m.x8 + 0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c23 = Constraint(expr=0.164046067482601*m.x8 - m.x34*(0.24997499749975*m.x3 + 0.375037503750375*m.x4 + 
                        0.375075018755251*m.x5 + 0.312562515629377*m.x6 + 0.234398432806401*m.x7 + 0.164046067482601*
                        m.x8 + 0.109331212792365*m.x9 + 0.0702562148313726*m.x10 + 0.0438881594251137*m.x11) == 0)

m.c24 = Constraint(expr=0.273519564077274*m.x8 - m.x35*(0.124962496249625*m.x4 + 0.249999992497001*m.x5 + 
                        0.312562515629377*m.x6 + 0.312593787516885*m.x7 + 0.273519564077274*m.x8 + 0.218793754368866*
                        m.x9 + 0.164062470437224*m.x10 + 0.117152290970943*m.x11) == 0)

m.c25 = Constraint(expr=0.273519564077274*m.x8 - m.x36*(0.0624624981253751*m.x5 + 0.156218732806561*m.x6 + 
                        0.234398432806401*m.x7 + 0.273519564077274*m.x8 + 0.273546935193458*m.x9 + 0.246192241674115*
                        m.x10 + 0.205139666893903*m.x11) == 0)

m.c26 = Constraint(expr=0.164046067482601*m.x8 - m.x37*(0.0312187515640633*m.x6 + 0.0937031062453125*m.x7 + 
                        0.164046067482601*m.x8 + 0.218793754368866*m.x9 + 0.246192241674115*m.x10 + 0.246216883075546*
                        m.x11) == 0)

m.c27 = Constraint(expr=0.0546382681253312*m.x8 - m.x38*(0.0156015671898445*m.x7 + 0.0546382681253312*m.x8 + 
                        0.109331212792365*m.x9 + 0.164062470437224*m.x10 + 0.205139666893903*m.x11) == 0)

m.c28 = Constraint(expr=0.00779610031479713*m.x8 - m.x39*(0.00779610031479713*m.x8 + 0.0312062456311301*m.x9 + 
                        0.0702562148313726*m.x10 + 0.117152290970943*m.x11) == 0)

m.c29 = Constraint(expr=0.0312062456311301*m.x9 - m.x40*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 
                        0.249999992497001*m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*
                        m.x8 + 0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c30 = Constraint(expr=0.109331212792365*m.x9 - m.x41*(0.24997499749975*m.x3 + 0.375037503750375*m.x4 + 
                        0.375075018755251*m.x5 + 0.312562515629377*m.x6 + 0.234398432806401*m.x7 + 0.164046067482601*
                        m.x8 + 0.109331212792365*m.x9 + 0.0702562148313726*m.x10 + 0.0438881594251137*m.x11) == 0)

m.c31 = Constraint(expr=0.218793754368866*m.x9 - m.x42*(0.124962496249625*m.x4 + 0.249999992497001*m.x5 + 
                        0.312562515629377*m.x6 + 0.312593787516885*m.x7 + 0.273519564077274*m.x8 + 0.218793754368866*
                        m.x9 + 0.164062470437224*m.x10 + 0.117152290970943*m.x11) == 0)

m.c32 = Constraint(expr=0.273546935193458*m.x9 - m.x43*(0.0624624981253751*m.x5 + 0.156218732806561*m.x6 + 
                        0.234398432806401*m.x7 + 0.273519564077274*m.x8 + 0.273546935193458*m.x9 + 0.246192241674115*
                        m.x10 + 0.205139666893903*m.x11) == 0)

m.c33 = Constraint(expr=0.218793754368866*m.x9 - m.x44*(0.0312187515640633*m.x6 + 0.0937031062453125*m.x7 + 
                        0.164046067482601*m.x8 + 0.218793754368866*m.x9 + 0.246192241674115*m.x10 + 0.246216883075546*
                        m.x11) == 0)

m.c34 = Constraint(expr=0.109331212792365*m.x9 - m.x45*(0.0156015671898445*m.x7 + 0.0546382681253312*m.x8 + 
                        0.109331212792365*m.x9 + 0.164062470437224*m.x10 + 0.205139666893903*m.x11) == 0)

m.c35 = Constraint(expr=0.0312062456311301*m.x9 - m.x46*(0.00779610031479713*m.x8 + 0.0312062456311301*m.x9 + 
                        0.0702562148313726*m.x10 + 0.117152290970943*m.x11) == 0)

m.c36 = Constraint(expr=0.00389531961090585*m.x9 - m.x47*(0.00389531961090585*m.x9 + 0.017542972627178*m.x10 + 
                        0.0438881594251137*m.x11) == 0)

m.c37 = Constraint(expr=0.017542972627178*m.x10 - m.x48*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 
                        0.249999992497001*m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*
                        m.x8 + 0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c38 = Constraint(expr=0.0702562148313726*m.x10 - m.x49*(0.24997499749975*m.x3 + 0.375037503750375*m.x4 + 
                        0.375075018755251*m.x5 + 0.312562515629377*m.x6 + 0.234398432806401*m.x7 + 0.164046067482601*
                        m.x8 + 0.109331212792365*m.x9 + 0.0702562148313726*m.x10 + 0.0438881594251137*m.x11) == 0)

m.c39 = Constraint(expr=0.164062470437224*m.x10 - m.x50*(0.124962496249625*m.x4 + 0.249999992497001*m.x5 + 
                        0.312562515629377*m.x6 + 0.312593787516885*m.x7 + 0.273519564077274*m.x8 + 0.218793754368866*
                        m.x9 + 0.164062470437224*m.x10 + 0.117152290970943*m.x11) == 0)

m.c40 = Constraint(expr=0.246192241674115*m.x10 - m.x51*(0.0624624981253751*m.x5 + 0.156218732806561*m.x6 + 
                        0.234398432806401*m.x7 + 0.273519564077274*m.x8 + 0.273546935193458*m.x9 + 0.246192241674115*
                        m.x10 + 0.205139666893903*m.x11) == 0)

m.c41 = Constraint(expr=0.246192241674115*m.x10 - m.x52*(0.0312187515640633*m.x6 + 0.0937031062453125*m.x7 + 
                        0.164046067482601*m.x8 + 0.218793754368866*m.x9 + 0.246192241674115*m.x10 + 0.246216883075546*
                        m.x11) == 0)

m.c42 = Constraint(expr=0.164062470437224*m.x10 - m.x53*(0.0156015671898445*m.x7 + 0.0546382681253312*m.x8 + 
                        0.109331212792365*m.x9 + 0.164062470437224*m.x10 + 0.205139666893903*m.x11) == 0)

m.c43 = Constraint(expr=0.0702562148313726*m.x10 - m.x54*(0.00779610031479713*m.x8 + 0.0312062456311301*m.x9 + 
                        0.0702562148313726*m.x10 + 0.117152290970943*m.x11) == 0)

m.c44 = Constraint(expr=0.017542972627178*m.x10 - m.x55*(0.00389531961090585*m.x9 + 0.017542972627178*m.x10 + 
                        0.0438881594251137*m.x11) == 0)

m.c45 = Constraint(expr=0.00194610043010828*m.x10 - m.x56*(0.00194610043010828*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c46 = Constraint(expr=0.00973926749128362*m.x11 - m.x57*(0.5*m.x2 + 0.5000500050005*m.x3 + 0.375037503750375*m.x4 + 
                        0.249999992497001*m.x5 + 0.156218732806561*m.x6 + 0.0937031062453125*m.x7 + 0.0546382681253312*
                        m.x8 + 0.0312062456311301*m.x9 + 0.017542972627178*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c47 = Constraint(expr=0.0438881594251137*m.x11 - m.x58*(0.24997499749975*m.x3 + 0.375037503750375*m.x4 + 
                        0.375075018755251*m.x5 + 0.312562515629377*m.x6 + 0.234398432806401*m.x7 + 0.164046067482601*
                        m.x8 + 0.109331212792365*m.x9 + 0.0702562148313726*m.x10 + 0.0438881594251137*m.x11) == 0)

m.c48 = Constraint(expr=0.117152290970943*m.x11 - m.x59*(0.124962496249625*m.x4 + 0.249999992497001*m.x5 + 
                        0.312562515629377*m.x6 + 0.312593787516885*m.x7 + 0.273519564077274*m.x8 + 0.218793754368866*
                        m.x9 + 0.164062470437224*m.x10 + 0.117152290970943*m.x11) == 0)

m.c49 = Constraint(expr=0.205139666893903*m.x11 - m.x60*(0.0624624981253751*m.x5 + 0.156218732806561*m.x6 + 
                        0.234398432806401*m.x7 + 0.273519564077274*m.x8 + 0.273546935193458*m.x9 + 0.246192241674115*
                        m.x10 + 0.205139666893903*m.x11) == 0)

m.c50 = Constraint(expr=0.246216883075546*m.x11 - m.x61*(0.0312187515640633*m.x6 + 0.0937031062453125*m.x7 + 
                        0.164046067482601*m.x8 + 0.218793754368866*m.x9 + 0.246192241674115*m.x10 + 0.246216883075546*
                        m.x11) == 0)

m.c51 = Constraint(expr=0.205139666893903*m.x11 - m.x62*(0.0156015671898445*m.x7 + 0.0546382681253312*m.x8 + 
                        0.109331212792365*m.x9 + 0.164062470437224*m.x10 + 0.205139666893903*m.x11) == 0)

m.c52 = Constraint(expr=0.117152290970943*m.x11 - m.x63*(0.00779610031479713*m.x8 + 0.0312062456311301*m.x9 + 
                        0.0702562148313726*m.x10 + 0.117152290970943*m.x11) == 0)

m.c53 = Constraint(expr=0.0438881594251137*m.x11 - m.x64*(0.00389531961090585*m.x9 + 0.017542972627178*m.x10 + 
                        0.0438881594251137*m.x11) == 0)

m.c54 = Constraint(expr=0.00973926749128362*m.x11 - m.x65*(0.00194610043010828*m.x10 + 0.00973926749128362*m.x11) == 0)

m.c55 = Constraint(expr=0.000972173680979933*m.x11 - 0.000972173680979933*m.x66*m.x11 == 0)

m.c56 = Constraint(expr=   0.5*m.x2 - 513*m.x12 - m.x67 + m.x77 == 0)

m.c57 = Constraint(expr=   0.75002500250025*m.x3 - 513*m.x13 - 41*m.x14 - m.x68 + m.x78 == 0)

m.c58 = Constraint(expr=   0.875037503750375*m.x4 - 513*m.x15 - 41*m.x16 - 100*m.x17 - m.x69 + m.x79 == 0)

m.c59 = Constraint(expr=   0.937537501874625*m.x5 - 513*m.x18 - 41*m.x19 - 100*m.x20 - 182*m.x21 - m.x70 + m.x80 == 0)

m.c60 = Constraint(expr=   0.968781248435937*m.x6 - 513*m.x22 - 41*m.x23 - 100*m.x24 - 182*m.x25 - 248*m.x26 - m.x71
                         + m.x81 == 0)

m.c61 = Constraint(expr=   0.984398432810156*m.x7 - 513*m.x27 - 41*m.x28 - 100*m.x29 - 182*m.x30 - 248*m.x31 - 167*m.x32
                         - m.x72 + m.x82 == 0)

m.c62 = Constraint(expr=   0.992203899685203*m.x8 - 513*m.x33 - 41*m.x34 - 100*m.x35 - 182*m.x36 - 248*m.x37 - 167*m.x38
                         - 89*m.x39 - m.x73 + m.x83 == 0)

m.c63 = Constraint(expr=   0.996104680389094*m.x9 - 513*m.x40 - 41*m.x41 - 100*m.x42 - 182*m.x43 - 248*m.x44 - 167*m.x45
                         - 89*m.x46 - 48*m.x47 - m.x74 + m.x84 == 0)

m.c64 = Constraint(expr=   0.998053899569892*m.x10 - 513*m.x48 - 41*m.x49 - 100*m.x50 - 182*m.x51 - 248*m.x52
                         - 167*m.x53 - 89*m.x54 - 48*m.x55 - 12*m.x56 - m.x75 + m.x85 == 0)

m.c65 = Constraint(expr=   0.99902782631902*m.x11 - 513*m.x57 - 41*m.x58 - 100*m.x59 - 182*m.x60 - 248*m.x61 - 167*m.x62
                         - 89*m.x63 - 48*m.x64 - 12*m.x65 - 2*m.x66 - m.x76 + m.x86 == 0)

m.c66 = Constraint(expr=   m.x2 + 2*m.x3 + 3*m.x4 + 4*m.x5 + 5*m.x6 + 6*m.x7 + 7*m.x8 + 8*m.x9 + 9*m.x10 + 10*m.x11
                         == 10000)

m.c67 = Constraint(expr=   m.x2 + m.x3 + m.x4 + m.x5 + m.x6 + m.x7 + m.x8 + m.x9 + m.x10 + m.x11 >= 513)

m.c68 = Constraint(expr=   m.x3 + m.x4 + m.x5 + m.x6 + m.x7 + m.x8 + m.x9 + m.x10 + m.x11 >= 41)

m.c69 = Constraint(expr=   m.x4 + m.x5 + m.x6 + m.x7 + m.x8 + m.x9 + m.x10 + m.x11 >= 100)

m.c70 = Constraint(expr=   m.x5 + m.x6 + m.x7 + m.x8 + m.x9 + m.x10 + m.x11 >= 182)

m.c71 = Constraint(expr=   m.x6 + m.x7 + m.x8 + m.x9 + m.x10 + m.x11 >= 248)

m.c72 = Constraint(expr=   m.x7 + m.x8 + m.x9 + m.x10 + m.x11 >= 167)

m.c73 = Constraint(expr=   m.x8 + m.x9 + m.x10 + m.x11 >= 89)

m.c74 = Constraint(expr=   m.x9 + m.x10 + m.x11 >= 48)

m.c75 = Constraint(expr=   m.x10 + m.x11 >= 12)

m.c76 = Constraint(expr=   m.x11 >= 2)

m.c77 = Constraint(expr= - m.x1 + m.x2 + m.x3 + m.x4 + m.x5 + m.x6 + m.x7 + m.x8 + m.x9 + m.x10 + m.x11 == 0)
