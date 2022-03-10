#from numba.npyufunc import parallel
from shapely.geometry import  MultiPoint
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

import pickle

# This function creates an array of pts inside Iceland a distance of 2000 meters from the shore.

def Get_pts_inside_of_country(polygon, distance = 4000, epsilon = 2000, side = 'right', mitre_limit = 5, tolerance = 350, preserve_topology = False, join_style = 3, testing = False, fast_run = True):
    if testing:
        with open('ISL_ADM2_polygons.p', 'rb') as handle:
            Poly_data = pickle.load(handle)
            #print(Poly_data.keys())
        polygon = Poly_data['MainLand']

    #if fast_run:
    #    distance = 20
    #    epsilon = 200
    #    tolerance = 1000
    #    preserve_topology = False

    # Change the Polygon to a LineString



    line = polygon.boundary

    #print(len(list(line)))
    # Sometimes threre are more then one string returned. Anyway. We only want the largest part of the country.
    # So we only use the largest polygon.
    print(type(line))
    if type(line).__module__ != 'shapely.geometry.linestring':
        index = 0
        max_lenght = 0
        for i, line_i in enumerate(line):
            print(np.array(line_i).shape)
            line_lenght = np.array(line_i).shape[0]
            if max_lenght < line_lenght:
                index = i
        line = list(line)[index]


    # Simplify the LineString by reducing the number of edges (This is necessary to be able to run the parallel_offset-function, we also do not need the high resolution)
    line = line.simplify(tolerance, preserve_topology = preserve_topology)

    print(type(line))

    # Get the LineStrings
    lines_offset = line.parallel_offset(distance = float(distance), side = side, mitre_limit = mitre_limit, join_style = join_style)

    try:
        lines_offset = list(lines_offset)
    except:
        lines_offset = [lines_offset]

    list_pts = []
    list_end_pts = []
    list_size = []
    list_dist = []
    for i, line in enumerate(lines_offset):
        if polygon.contains(line):
            array_pts = np.array(line.xy).T


            xy1 = array_pts[0:len(array_pts)-2,:]
            xy2 = array_pts[1:len(array_pts)-1,:]
            dist = np.linalg.norm(xy2-xy1, axis = 1, ord = 2)

            if (array_pts[0,:] == array_pts[-1,:]).all():
                list_dist.append(dist[0:len(dist) - 2])
                list_pts.append(array_pts[0:len(array_pts) - 2,:])
            else:
                list_dist.append(dist)
                list_pts.append(array_pts)

    new_list = []
    for i in range(0, len(list_pts)):
        pts = list_pts[i]
        pts_new = list(map(tuple,np.array(rdp(pts, epsilon))))
        new_list.extend(pts_new)
    MLS = MultiPoint(new_list)

    if testing:
        fig, ax = plt.subplots(1,1)

        main_land_geoplot = gpd.GeoSeries(polygon)
        main_land_geoplot.plot(ax = ax, color = 'blue')

        offset_geoplot = gpd.GeoSeries(MLS)
        offset_geoplot.plot(ax = ax, color = 'red')

        plt.show()
    array = np.concatenate(list_pts)
    data = {'Array': array}
    return data


def rdp(points, epsilon):  # https://towardsdatascience.com/simplify-polylines-with-the-douglas-peucker-algorithm-ac8ed487a4a1
    # get the start and end points
    start = np.tile(np.expand_dims(points[0], axis=0), (points.shape[0], 1))
    end = np.tile(np.expand_dims(points[-1], axis=0), (points.shape[0], 1))

    # find distance from other_points to line formed by start and end
    dist_point_to_line = np.abs(np.cross(end - start, points - start, axis=-1)) / np.linalg.norm(end - start, axis=-1)
    # get the index of the points with the largest distance
    max_idx = np.argmax(dist_point_to_line)
    max_value = dist_point_to_line[max_idx]


    result = []
    if max_value > epsilon:
        partial_results_left = rdp(points[:max_idx+1], epsilon)
        result += [list(i) for i in partial_results_left if list(i) not in result]
        partial_results_right = rdp(points[max_idx:], epsilon)
        result += [list(i) for i in partial_results_right if list(i) not in result]
    else:
        result += [points[0], points[-1]]

    return result


#Get_pts_inside_of_country(polygon = None, testing = True)
#Get_pts_inside_of_country(
#    polygon = None,
#    distance = 5000,
#    epsilon=20000,
#    mitre_limit = 20,
#    tolerance = 800,
#    preserve_topology = True,
#    join_style = 3,
#    testing = True
#)

#Get_pts_inside_of_country(
#    polygon = None,
#    distance = 7000,
#    epsilon=8000,
#    mitre_limit = 1,
#    tolerance = 500,
#    preserve_topology = True,
#    join_style = 3,
#    testing = True
#)

#Get_pts_inside_of_country(
#    polygon = None,
#    distance = 9000,
#    epsilon=10000,
#    mitre_limit = 1,
#    tolerance = 500,
#    preserve_topology = True,
#    join_style = 3,
#    testing = True
#)

#Get_pts_inside_of_country(
#    polygon = None,
#    distance = 7500,
#    epsilon=20000,
#    mitre_limit = 4,
#    tolerance = 5000,
#    preserve_topology = True,
#    join_style = 3,
#    testing = True
#)
