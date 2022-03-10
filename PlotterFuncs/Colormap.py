
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import numpy as np


def Get_ColorMap(z, element='SWH', increase_resolution_const = 1):
    # If element = 'unspecified'. Then determine what element should be
    element = Get_Element(z, element)

    # Get Color Codes used by Vegagerdin
    colors_values, colors_RGB = Color_Codes(element)

    # Change the values into intervals
    color_intervals, color_RGB = create_intervals_for_color_codes(colors_values, colors_RGB)

    # add contour levels. If increase_resolution_const = 1. Then it will be the same. If increase_resolution_const=2 then we will double the contour levels and split each interval in two intervals
    if increase_resolution_const != 1:
        color_intervals, color_RGB = Add_Contour_Levels(color_intervals, color_RGB, increase_resolution_const)

    # Create the <matplotlib.colors.ListedColormap> object
    my_cmap, my_norm = create_colormap(color_intervals, color_RGB)
    return my_cmap, my_norm

# This function is only used if the element is not specified
def Get_Element(z, element):

    bool_element = (element[0:3] == 'SWH'or element[0:3] == 'MWD' or element[0:3] == 'MWP')
    bool_SWH = (element[0:3] == 'SWH' or element == 'unspecified')
    bool_MWD = (element[0:3] == 'MWD' or element == 'unspecified')
    bool_MWP = (element[0:3] == 'MWP' or element == 'unspecified')
    assert bool_element or element == 'unspecified', 'Warning, wrong input element. There is no colormap for an element, named [{}]'.format(element)
    if element == 'unspecified':
        z_max = np.nanmax(z[z < 500])
        SWH35_max_val = 3.625
        SWH7_max_val  = 7.25
        MWP_max_val   = 16
        MWD_max_val   = 360
        if z_max <= SWH35_max_val and bool_SWH:
            element = 'SWH35'
        elif z_max <= SWH7_max_val and bool_SWH:
            element = 'SWH7'
        elif z_max <= MWP_max_val and bool_MWP:
            element = 'MWP'
        elif z_max <= MWD_max_val and bool_MWD:
            element = 'MWD'
        else:
            print('\n#############')
            print('## WARNING ##')
            print('#############')
            print('Element input for plotting: -> ', element)
            print('z_max value: -> ', z_max)
            print('we have no idea what element this is supposed to be.')
            assert element == '?'
    else:
        if element[:3] == 'MWD':
            element = 'MWD'
        elif element[:3] == 'MWP':
            element = 'MWP'
        elif element == 'SWH':
            element = 'SWH35'

    return element



def create_intervals_for_color_codes(colors_values, colors_RGB):
    colors_RGB = [x for _,x in sorted(zip(colors_values, colors_RGB))]
    colors_values.sort()

    interval = colors_values[1] - colors_values[0]
    color_intervals = np.array(colors_values)

    color_RGB = np.array(colors_RGB)/256
    assert color_RGB.shape[1] == 3
    color_RGB = np.concatenate([color_RGB, np.ones(shape = (color_RGB.shape[0],1))], axis = 1)

    return color_intervals, color_RGB

def Add_Contour_Levels(color_intervals, color_RGB, increase_const):
    nr_levels = color_RGB.shape[0]

    final_color_intervals = color_intervals[-1]
    final_color_RGB = color_RGB[-1,:]

    color_intervals = color_intervals[:nr_levels-1]
    color_RGB = color_RGB[:nr_levels-1,:]

    new_nr_levels = (nr_levels-1)*increase_const

    color_intervals_out = np.zeros(shape = (new_nr_levels,))
    color_RGB_out = np.zeros(shape = (new_nr_levels, 4))

    intervals_lenght = (color_intervals[1] - color_intervals[0])/2

    color_1 = color_RGB[0,:]
    color_2 = color_RGB[1,:]

    color_intervals_out[0] = 0
    color_RGB_out[0,:] = color_1 - (color_2 - color_1)/2

    # 0 -> new color
    # 1 -> color 0
    # 2 -> new color
    # 3 -> color 1
    # 4 -> new color
    # 5 -> color 2
    # 6 -> new color
    # 7 -> color 3
    # 8 -> new color
    # 9 -> color 4
    for i in range(1,new_nr_levels-1):
        color_intervals_out[i] = intervals_lenght*i
        if i%2 == 1:
            color_index = i//2
            color_RGB_out[i,:] = color_RGB[color_index,:]
        else:
            first_color = color_RGB[i//2 - 1,:]
            second_color = color_RGB[i//2,:]
            change = second_color - first_color
            color_RGB_out[i,:] = first_color + change/2

    color_intervals_out[-1] = final_color_intervals
    color_RGB_out[-1] = final_color_RGB

    return color_intervals_out, color_RGB_out

def create_colormap(colors_intervals, color_RGB):
    lenght = len(colors_intervals)
    my_cmap = matplotlib.colors.ListedColormap(color_RGB)
    my_cmap.set_over(color = 'white')
    my_norm = matplotlib.colors.BoundaryNorm(colors_intervals[:lenght-1], colors_intervals.shape[0]-1)
    return my_cmap, my_norm

def Color_Codes(element):
    color_codes = {
            "SWH35":{
                "Litakodi": [
                    (255,   255, 255),
                    (255,   255, 255),
                    (255,   0,   0),
                    (255,  85,   0),
                    (255, 170,   0),
                    (255, 255,   0),
                    (170, 255,   0),
                    ( 85, 255,   0),
                    (  0, 255,   0),
                    (  0, 226,  55),
                    (  0, 196, 110),
                    (  0, 166, 166),
                    (  0, 111, 195),
                    (  0,  56, 225),
                    (  0,   0, 255),
                    ( 42,   0, 213),
                    ( 85,   0, 171),
                    (128,   0, 128),
                ],
                "Gildi": [
                    1000000,
                    4.00,
                    3.50,
                    3.25,
                    3.00,
                    2.75,
                    2.50,
                    2.25,
                    2.00,
                    1.75,
                    1.50,
                    1.25,
                    1.00,
                    0.75,
                    0.50,
                    0.25,
                    0.00,
                    -9999
                ]
            },
            "SWH7":{
                "Litakodi": [
                    (255,   255, 255),
                    (255,   255, 255),
                    (255,   0,   0),
                    (255,  85,   0),
                    (255, 170,   0),
                    (255, 255,   0),
                    (170, 255,   0),
                    ( 85, 255,   0),
                    (  0, 255,   0),
                    (  0, 226,  55),
                    (  0, 196, 110),
                    (  0, 166, 166),
                    (  0, 111, 195),
                    (  0,  56, 225),
                    (  0,   0, 255),
                    ( 42,   0, 213),
                    ( 85,   0, 171),
                    (128,   0, 128)
                ],
                "Gildi": [
                    1000000,
                    100,
                    7,
                    6.5,
                    6.0,
                    5.5,
                    5.0,
                    4.5,
                    4.0,
                    3.5,
                    3.0,
                    2.5,
                    2.0,
                    1.5,
                    1.0,
                    0.5,
                    0,
                    -9999
                ]
            },
            "MWP":{
                "Litakodi": [
                    (255,   0,   0),
                    (255,  85,   0),
                    (255, 170,   0),
                    (255, 255,   0),
                    (170, 255,   0),
                    ( 85, 255,   0),
                    (  0, 255,   0),
                    (  0, 226,  55),
                    (  0, 196, 110),
                    (  0, 166, 166),
                    (  0, 111, 195),
                    (  0,  56, 225),
                    (  0,   0, 255),
                    ( 42,   0, 213),
                    ( 85,   0, 171),
                    (128,   0, 128),
                    (  0,   0,   0)
                ],
                "Gildi": [
                    1000000,
                    20,
                    15,
                    14,
                    13,
                    12,
                    11,
                    10,
                    9,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                    1,
                    0,
                    -9999
                ]
            },
            "MWD":{
                "Litakodi": [
                    (128,   0, 128),
                    (255,   0,   0),
                    (255,  85,   0),
                    (255, 170,   0),
                    (255, 255,   0),
                    (170, 255,   0),
                    ( 85, 255,   0),
                    (  0, 255,   0),
                    (  0, 226,  55),
                    (  0, 196, 110),
                    (  0, 166, 166),
                    (  0, 111, 195),
                    (  0,  56, 225),
                    (  0,   0, 255),
                    ( 42,   0, 213),
                    ( 85,   0, 171),
                    (128,   0, 128),
                    (255,   0,   0)
                ],
                "Gildi": [
                    1000000,
                    360.0,
                    337.5,
                    315.0,
                    292.5,
                    270.0,
                    247.5,
                    225.0,
                    202.5,
                    180.0,
                    157.5,
                    135.0,
                    112.5,
                    90.0,
                    67.5,
                    45.0,
                    22.5,
                    0,
                    -9999
                ]
            }
    }

    colors_values = color_codes[element]['Gildi']
    colors_RGB = color_codes[element]['Litakodi']
    return colors_values, colors_RGB

