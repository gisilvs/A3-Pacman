import matplotlib.pyplot as plt
import numpy as np

food_eaten=np.loadtxt('food_eaten.txt',dtype=int)
food_returned=np.loadtxt('food_returned.txt',dtype=int)
score=np.loadtxt('score.txt',dtype=int)


food_eaten_avg=[]
food_returned_avg=[]
score_avg=[]
for i in range(0,len(food_eaten)-10,10):
    food_eaten_avg.append(np.mean(food_eaten[i:i+10]))
    food_returned_avg.append(np.mean(food_returned[i:i+10]))
    score_avg.append(np.mean(score[i:i+10]))

a=0
# Get current size
fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]
print("Current size:", fig_size)

# Set figure width to 12 and height to 9
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

plt.plot(food_eaten_avg)
plt.title('Mean of eaten food during training')
plt.xlabel('Every 10 epochs')
plt.ylabel('Units of food eaten')
plt.show()
plt.plot(food_returned_avg, 'r')
plt.title('Mean of food returned during training')
plt.xlabel('Every 10 epochs')
plt.ylabel('Units of food returned')
plt.show()
plt.plot(score_avg, 'g')
plt.title('Mean score during training')
plt.xlabel('Every 10 epochs')
plt.ylabel('Score')
plt.show()