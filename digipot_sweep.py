import moku
import numpy as np
import AD


# define test frequencies
frequencies = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] * 1e3
# define test voltage
voltage = 0.2
# pre-made save array
optimal_resistance = []
for frequency in frequencies:
    # variant 1: sweep over all resistance values,
    # save all values, choose best resistances in postproc
    # variant 2: binary search for best value

    # set resistance value

    # prepare measurement

    # run measurement

    # get data out

    # estimate

    # decide next step

    # save if optimal

# save all data