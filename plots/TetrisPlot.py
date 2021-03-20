import json


with open('DQNmeanloss.txt') as f:
    DQNmeanloss = json.load(f)
    
with open('DQNsd.txt') as f:
    DQNsd = json.load(f)
    
with open('DQNScore.txt') as f:
    DQNScore = json.load(f)
    
with open('DRQNmeanloss.txt') as f:
    DRQNmeanloss = json.load(f)
    
with open('DRQNsd.txt') as f:
    DRQNsd = json.load(f)

with open('DRQNScore.txt') as f:
    DRQNScore = json.load(f)



import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.plot(DQNScore,label='DQN', color = 'red')
plt.plot(DRQNScore,label='DRQN', color = 'green')
plt.plot()
plt.ylabel('Score', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.title('Scores', fontsize=20)
plt.legend(fontsize=20)
plt.show()


x = range(len(DQNmeanloss))


aa = []
for i in range(len(x)):
    z = x[i]*50
    aa.append(z)


plt.figure(figsize=(40,20))

plt.errorbar(aa, DQNmeanloss, DQNsd,label = "DQN", color = 'red')
plt.errorbar(aa, DRQNmeanloss, DRQNsd, label = "DQN", color = 'green')
#plt.errorbar
plt.plot()
plt.ylabel('Standard Deviation', fontsize=20)
plt.xlabel('Every 50 Epochs', fontsize=20)
plt.title('Standard Deviation', fontsize=20)
plt.legend(fontsize=20)
plt.show()
