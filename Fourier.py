#%% Libraries
import numpy as np
import matplotlib.pyplot as plt

#%% import signal
time1, data1 = np.loadtxt("Fourier/data.txt", unpack = True, usecols=(0,1))
time = time1[1::10]
data = data1[1::10]
print(time, data)
print(len(data))
T = (time[2]-time[1])
N = round(2*np.pi/T) - 1
print(N)

#%% find Cn


C = []
for k in range (N):
    Ci = []
    for i in range (len(data)):
        A = data[i]*np.exp(-1j*k*time[i])*T/(2*np.pi)
        Ci.append(A)
    C.append(np.sum(Ci))




#%% define f(x)

n = np.arange(len(C)) + 1

def f(x):
    func = sum(C*np.exp(1j*n*x))
    return func



out = []

for l in range (len(time)):
    out.append(f(time[l]))

print(sum(data - out))

#%% plotting
plt.plot(time1, data1)
plt.plot(time, data, 'g-')
plt.show()
plt.plot(time1, data1)
plt.plot(time, out, 'r-')
plt.show()



#%% putting into txt

tf = []
for i in range (len(time)):
    for j in range (99):
        tf.append(time[i])

of = []
for i in range (len(time)):
    for j in range (99):
        of.append(out[i])

#final = np.column_stack((np.real(tf), np.real(of)))
final = np.column_stack((np.real(time), np.real(out)))


np.savetxt('process.txt', final)


# %%
