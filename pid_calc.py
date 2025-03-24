import numpy as np

# setup one: comp bridge, opamp off
Gp = 3.3
Fd = 374.5e3
# result: 2khz suppressed from 47uV to 17uV

# setup two: pure membrane
Gp = 4.8
Fd = 729.8e3

fs = 44100
T = 3.0
print(fs * T)
