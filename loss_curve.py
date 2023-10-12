import re
import matplotlib.pyplot as plt

filepath='/home/zl437/rds/hpc-work/gector/2e-6.txt'

f = open(filepath, "r")
lines = f.readlines()

loss = []
ece = []
auc = []
f = []

for l in lines:
    if l.startswith('Test'):
        num = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", l)
        loss.append(float(num[0]))
    elif l.startswith('ece'):
        num = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", l)
        ece.append(float(num[0]))
        auc.append(float(num[1]))
        f.append(float(num[5]))

plt.subplot(2, 1, 1)
plt.plot(range(len(loss)),loss)
plt.plot(range(len(auc)),auc)
plt.plot(range(len(f)),f)
plt.ylabel('Value')
plt.legend(['Loss', 'AUC', 'F 0.5'])

plt.subplot(2,1,2)
plt.plot(range(len(ece)),ece)
    
plt.xlabel('Epochs')
plt.ylabel('ECE')
plt.legend(['ECE'])

plt.savefig("loss.png")
