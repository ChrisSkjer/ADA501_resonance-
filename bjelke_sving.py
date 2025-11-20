import numpy as np

l=2
E=10000
I=1/12*48*198
m= 1000
#Egenfrekvens etter NS-EN 1995-1-1

f1 = np.pi/(2*l**2)*np.sqrt(E*I/m)

vinkel1 = f1

#egenfrevena 1dof

vinkel2 = 8.76/(l**1.5)*np.sqrt(E*I/m)/(2*np.pi)

print(vinkel1,vinkel2)

