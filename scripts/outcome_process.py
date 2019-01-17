import numpy as np

outcome = np.load('outcome.npy')
outcome[outcome == 1] = 0
outcome[outcome == 2] = 1
outcome[outcome == 4] = 2
outcome[outcome == 8] = 3
outcome[outcome == 16] = 4

np.save('outcome_ohv.npy', outcome)
