import numpy as np
import matplotlib.pyplot as plt

#betrakter ligningen m*x'' + c*x' + k*x = 0
#om c^2-4mk < 0 får vi svinging, tilsvarer zeta < 1,0

#start betingelser 
x0 = 1
v0 = 0

t = np.linspace(0,10,1000)
m = 1
c = 1
k = 20
D =1
E = 1

if c*c - 4*m*k< 0:
    x_full = np.exp(-c/(2*m)*t) * (D*np.cos(np.sqrt(4*m*k-c**2)/(2*m)*t) + D*np.sin(np.sqrt(4*m*k-c**2)/(2*m)*t))
    # x_test er for å vise frekvensen med demping uten å ta med dempingsleddet. 
    x_test = (D*np.cos(np.sqrt(4*m*k-c**2)/(2*m)*t) + D*np.sin(np.sqrt(4*m*k-c**2)/(2*m)*t))
    print("svinger")
else:
    r1 = (-c+np.sqrt(c**2-4*m*k))/(2*m)
    r2 = (-c-np.sqrt(c**2-4*m*k))/(2*m)
    x_full = D*np.exp(r1*t)+E*np.exp(r2*t)
    print("fult dempet")
omega = np.sqrt(k/m)

#hvordan bygget svinger uten demping
x = D*np.cos(omega*t)+E*np.sin(omega*t) 

plt.plot(t,x)
plt.plot(t,x_full)
plt.show()