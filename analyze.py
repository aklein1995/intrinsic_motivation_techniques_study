
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')



fpath = 'storage/MultiRoomN7S4/ICM/MN4S7_im01_ent00001/'
data = pd.read_csv(fpath + 'log.csv')
data = data.drop([1,3,5,7,9,11,13,15,17],axis=0) # drop columns with headers

# select just avg return
avg_return = data['avg_return'].astype(dtype=float)
frames = data['frames'].astype(dtype=int)


plt.plot(frames,avg_return)
plt.legend()
plt.xlabel('Number of frames')
plt.ylabel('Average Return')
plt.ylim([0,1])
plt.show()
