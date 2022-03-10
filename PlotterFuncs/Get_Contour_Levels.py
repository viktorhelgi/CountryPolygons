import numpy as np

def Get_Contour_Levels(z, levels = 80):   
    z_without_nan_values = z[~np.isnan(z)]
    z_without_9999_values = z_without_nan_values[(z_without_nan_values < 9999)]
    min_val = min(z_without_9999_values)-1
    max_val = max(z_without_9999_values)

    MaxWaveHeight = max_val+max_val/4
    MinWaveHeight = min_val
    NumberOfLevels = levels
    OneInterval = (MaxWaveHeight - MinWaveHeight)/NumberOfLevels
    Levels = list(np.arange(MinWaveHeight, MaxWaveHeight + OneInterval, OneInterval))
    Levels.pop(len(Levels)-1)
    Levels.pop(len(Levels)-1)
    return Levels
