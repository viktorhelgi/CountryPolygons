
#################
# Local modules #
#################
# Main Functions
from .GetShapefile import Get_Shapefile
#from .Create_Polygons import Create_Polygons
# Helper Functions
from .Project_data import Project_data
from .Polygon_Array_Transformation import  poly_arr_tr
from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, MultiPolygon, Polygon

import numpy as np
import math

####################
# External Modules #
####################
from typing import List, Dict, Any, TypedDict

################################################################################################################################################
# Code Usage:                                                                                                                                  #
################################################################################################################################################
# This file is used to create pickle-files from shp-files. The variable GeoBoundary_file is the path of a High-Res shp file of country.      #
# The shp file was obtained from the following website: https://www.geoboundaries.org/downloadFull.html.                                       #
# An API can be used to get all the files needed. However, note that the 'High Precision Unstandardized' files include Islands. The            #
# The 'High Precision Standardized' do not include Islands                                                                                     #
# This code will only be needed if we will need polygon-files for other countries than Island.                                                 #
# The Polygon files are used when the LargMaps are created. The are used to change the values of the cells where countries are located.        #
# Using these polygon files and the function 'def country_Insert_values' this can be done very quickly.                                        #
# It is probably best to create polygons which are triangular. However, for simplicity we created rectangular polygons.                        #
################################################################################################################################################
class OutputDict(TypedDict):
    polygons:List[np.ndarray]
    country_borders:List[np.ndarray]


def create_country_polygons(
    GeoBoundary_file,
    plot_polygons = False,
    proj_target = '+proj=utm +zone=27',
    lenght_of_interval = 10000
) -> OutputDict:
    print('\nCCPP: def create_country_polygons_process()')
    info = GeoBoundary_file.split('/')
    country_name = info[len(info)-2]

    print('CCPP: input-file: ', GeoBoundary_file)
    print('CCPP: Country: ', country_name)

    # ------------------------------------------------------------------------------------------------
    # Get polygons from shapefile as array, and get the projection
    country_boundaries_dat = Get_Shapefile(
        file = GeoBoundary_file
    )
    polygons = country_boundaries_dat['polygons']
    proj_wkt = country_boundaries_dat['proj_wkt']
    # ------------------------------------------------------------------------------------------------
    # Project the array to the following coordinates
    country_boundaries_dat = Project_data(
        data = polygons,
        proj_wkt_source = proj_wkt,
        proj_wkt_target = proj_target)
    polygons = country_boundaries_dat['polygons']
    proj_wkt = country_boundaries_dat['proj_wkt']
    # ------------------------------------------------------------------------------------------------
    # Create Multipolygons for country, and create a single polygon for the main land
    country_polygons_dat = Create_Polygons(
        polygons = polygons,
        nr_of_intervals = lenght_of_interval,
    )
    return country_polygons_dat

    #polygons = country_polygons_dat['polygons']
    #main_land = country_polygons_dat['main_land']
    #boundaries = country_polygons_dat['boundaries']
    #print(type(polygons))
    #print(type(main_land))
    #print(type(boundaries))

    #file = '{}xx.p'.format(filename)
    #with open(file, 'wb') as handle:
    #    pickle.dump(country_polygons_dat, handle)

    # ------------------------------------------------------------------------------------------------
    # Change the projection of boundaries to Pseudo-Mercator (Open-street-maps projection)
    #if type(list(boundaries)[0]).__module__ == 'shapely.geometry.polygon':
    #    boundaries = poly_arr_tr(boundaries)
    #boundaries = Project_data(
    #    data = boundaries,
    #    proj_wkt_source= proj_target,
    #    proj_wkt_target= 4326
    #)
    ## ------------------------------------------------------------------------------------------------
    #data = {
    #    'Polygons': polygons,
    #    #'Main_Land': main_land,
    #    #'boundaries': boundaries,
    #    'proj': 4326
    #}
    #try:
    #    info = file.split('/')
    #    country_name = 'Norway'
    #    filename_out = '{}_polygonsxx.p'.format(country_name)
    #    with open(filename_out, 'wb') as handle:
    #        pickle.dump(data, handle)
    #except:
    #    print('CCPP: Error while saving pickle')
    #else:
    #    print('CCPP: Pickle saved successfully')


# Local modules


def Create_Polygons(
    polygons,
    nr_of_intervals = 100,
    proj_target='+proj=utm +zone=27'
) -> OutputDict:
    print('\n- CP: def Create_Polygons():')
    '''Inputs:
    '''
    if type(list(polygons)[0]).__module__ != 'numpy':
         raise NotImplementedError('\nFunction: "Create_Polygons(polygons, ...)"\nProblem:polygons must be an array or a list of np.ndarray\'s')

    #Multipolygon: MultiPolygon = MultiPolygon()
    boundaries:MultiPolygon = poly_arr_tr(obj_in = polygons, output_type = 'multipolygon')

    print('  CP: Nr of Polygons before combination: ', len(boundaries.geoms))
    Multipolygon_only_Borders:MultiPolygon = unary_union(boundaries) # Returns the union of a sequence of geometries
    print('  CP: Nr of Polygons after combination: ', len(Multipolygon_only_Borders.geoms))


    # Get MainLand Polygon
    temp_Multipoly = Multipolygon_only_Borders.geoms
    max_index = 0
    max_area = 0
    x_min = math.inf
    y_min = math.inf
    x_max = -math.inf
    y_max = -math.inf

    for i, poly in enumerate(temp_Multipoly):
        bounds = poly.bounds
        if max_area < poly.area:
            max_area = poly.area
            max_index = i
        if bounds[0] < x_min:
            x_min = bounds[0]
        if bounds[1] < y_min:
            y_min = bounds[1]
        if x_max < bounds[2]:
            x_max = bounds[2]
        if y_max < bounds[3]:
            y_max = bounds[3]

    main_land:Polygon = temp_Multipoly[max_index]
    bounds = Multipolygon_only_Borders.bounds

    #--------------------------------------------------------------------------
    print('  CP: Create Grid')
    grid, box_area = create_grid(
        bounds = bounds,
        nr_of_intervals = nr_of_intervals
    )
    #--------------------------------------------------------------------------
    print('  CP: Create Polygons')
    polygons_out:List[np.ndarray] = Creat_Polygons_Procedure(
        grid = grid,
        Polygons = Multipolygon_only_Borders,
        box_area = box_area,
        proj_target = proj_target
    )
    print(len(polygons_out))

    if type(list(polygons)[0]).__module__ == 'numpy':
        Multipolygon = poly_arr_tr(obj_in = polygons_out, output_type = 'multipolygon')
    Multipolygon_only_Borders = unary_union(Multipolygon)
    polygons_borders = poly_arr_tr(obj_in=Multipolygon_only_Borders, output_type='numpy')

    print(type(polygons_borders))
    print(type(polygons_borders[0]))
    #nr_of_polygons = len(list(Multipolygon_out.geoms))

    #print('      nr of polygons: ', nr_of_polygons)

    #data:OutputDict = {
    #    'boundaries' : boundaries,   # This is simply all the boundaries of the islands and the mainland.
    #    'polygons': polygons_out,            # This is a list of polygons which will be used when inserting countries into maps
    #    'main_land': main_land,          # This is the main land polygon. It is also included in the the list::Multipolygon_only_Borders
    #}

    data:OutputDict = {
        'polygons':polygons_out,
        'country_borders':polygons_borders
    }
    return data

def Creat_Polygons_Procedure(
        grid,
        Polygons,
        box_area,
        proj_target) -> List[np.ndarray]:

    print('      - CPP: def Creat_Polygons_Procedure')


    polygons_out = []
    # Find Polygons which will be cut into smaller polygons.
    polygons_large = []
    polygons_large_size = []
    polygons_small = []

    for i in range(len(list(Polygons.geoms))):
        polygon = Polygons.geoms[i]
        if box_area <= polygon.area:
            polygons_large.append(polygon)
            polygons_large_size.append(polygon.area)
        else:
            polygons_small.append(polygon)
            array = np.array(polygon.boundary.coords)
            data = Project_data(data = [array], proj_wkt_source=proj_target, proj_wkt_target='+proj=longlat +datum=WGS84 +no_defs')
            poly_out = data['polygons'][0]
            polygons_out.append(poly_out)
    #polygons_out.extend(polygons_small)

    polygons_large = [x for _,x in sorted(zip(polygons_large_size, polygons_large))]
    nr_of_large_polygons = len(polygons_large)
    print('        CPP: Splitting the Large Polygons (The smallest of the large Polygons are splitted first)')
    str1 = ''
    str2 = ''
    for i in range(0, nr_of_large_polygons):
        perc_i = round((i/nr_of_large_polygons)*100,2)
        str1 = '        CPP: {}%'.format(perc_i)
        polygon = polygons_large[i]

        union = polygon.boundary.union(grid)
        unioned = list(polygonize(union))

        nr_of_polys_in_unioned = len(unioned)
        for j, poly in enumerate(unioned):
            if poly.representative_point().within(polygon):
                try:
                    array = np.array(poly.boundary.coords)
                    data = Project_data(data = [array], proj_wkt_source=proj_target, proj_wkt_target='+proj=longlat +datum=WGS84 +no_defs')
                    poly_out = data['polygons'][0]
                    polygons_out.append(poly_out)
                except:
                    raise Exception("Code failed")
            perc_j = int((j/nr_of_polys_in_unioned)*100)
            if perc_j%2 == 0:
                str2 = '\t {}%'.format(perc_j)
                print(str1 + str2, end = '\r')
    print(str1 + str2)
    return polygons_out

    #Multipolygon_out = MultiPolygon(polygons_out)
    #return Multipolygon_out

def create_grid(bounds, nr_of_intervals = None):
    x_line = np.arange(bounds[0], bounds[2], nr_of_intervals)
    y_line = np.arange(bounds[1], bounds[3], nr_of_intervals)
    n = y_line.shape[0]
    m = x_line.shape[0]

    x_lenght = x_line.max() - x_line.min()
    y_lenght = y_line.max() - y_line.min()
    box_area = (x_lenght/m)*(y_lenght/n)

    x_list = [x_line, x_line[::-1]]
    y_list = [y_line, y_line[::-1]]

    if len(y_line)%2 == 0:
        b = 1
    else:
        b = 0


    x_grid = [(x,y) for i, y in enumerate(y_line) for x in x_list[int(i%2)]]
    y_grid = [(x,y) for i, x in enumerate(x_line) for y in y_list[int((i+b)%2)]]
    grid = LineString(x_grid + y_grid)

    return grid, box_area
