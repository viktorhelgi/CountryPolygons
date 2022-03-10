#from functions.DataClasses import GridCoords, PolyCoords
#from shapely.geometry import MultiPoint
#from functions.ShapelyNumpyTransformation import poly_arr_tr
#import numpy as np
#
#
#
#
#def convexhull(grid:GridCoords) -> PolyCoords:
#    points = []
#    x,y = grid.x, grid.y
#    n,m = x.shape
#    x_bound = np.concatenate([x[:,0], x[:,m-1], x[0,:], x[n-1,:]])
#    y_bound = np.concatenate([y[:,0], y[:,m-1], y[0,:], y[n-1,:]])
#
#    for i,j in zip(
#            x_bound,
#            y_bound):
#        points.append((i,j))
#
#    multi_point = MultiPoint(points)
#    polygon_hull = multi_point.convex_hull
#
#    array:np.ndarray = poly_arr_tr(polygon_hull)
#    return PolyCoords(
#        x= array[:,0],
#        y = array[:,1],
#        proj = grid.proj
#    )
#
