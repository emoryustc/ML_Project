import numpy as np

outcome = np.load('outcome.npy')
outcome[outcome == 1] = 0
outcome[outcome == 2] = 1
outcome[outcome == 4] = 2
outcome[outcome == 8] = 3
outcome[outcome == 16] = 4

np.save('outcome_ohv.npy', outcome)

new_outcome = np.zeros(((outcome.shape[0]), 5))
for i in range(outcome.shape[0]):
    if outcome[i] == 0:
        new_outcome[i, 0] = 1
    elif outcome[i] == 1:
        new_outcome[i, 1] = 1
    elif outcome[i] == 2:
        new_outcome[i, 2] = 1
    elif outcome[i] == 3:
        new_outcome[i, 3] = 1
    elif outcome[i] == 4:
        new_outcome[i, 4] = 1

np.save('outcome_ohv_r5.npy', outcome)
