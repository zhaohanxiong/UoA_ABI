import numpy as np


TP = 98.0
TN = 91.2
FP = 8.8
FN = 2.0

precision = TP/(TP + FP)
recall = TP/(TP + FN)

print(2 * (recall * precision)/(recall + precision))

#(92.8% vs 89.6%) as opposed to SR signals (98.1% vs 97.4%)
