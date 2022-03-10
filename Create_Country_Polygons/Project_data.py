import numpy as np
from osgeo import osr

def Project_data(data, proj_wkt_source, proj_wkt_target):
    srs_source = osr.SpatialReference()
    srs_target = osr.SpatialReference()

    if type(proj_wkt_source) == str:
        if proj_wkt_source[0] == '+': # We were given a projection formated with PROJ4. We will change it to WKT
            srs = osr.SpatialReference()
            srs.ImportFromProj4(proj_wkt_source)
            proj_wkt_source = srs.ExportToWkt()
        srs_source.ImportFromWkt(proj_wkt_source)

    elif type(proj_wkt_source) == int:
        srs_source.ImportFromEPSG(proj_wkt_source)

    else:
        ERROR = False
        assert ERROR, "proj_wkt_source, isn't a string nor an int"

    if type(proj_wkt_target) == str:
        if proj_wkt_target[0] == '+': # We were given a projection formated with PROJ4. We will change it to WKT
            srs = osr.SpatialReference()
            srs.ImportFromProj4(proj_wkt_target)
            proj_wkt_target = srs.ExportToWkt()
        srs_target.ImportFromWkt(proj_wkt_target)

    elif type(proj_wkt_target) == int:
        srs_target.ImportFromEPSG(proj_wkt_target)

    else:
        ERROR = False
        assert ERROR, "proj_wkt_target, isn't a string nor an int"

    srs_source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) #https://gis.stackexchange.com/questions/364943/gdal-3-0-4-invalid-coordinate-transformation-result
    srs_target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) #https://gis.stackexchange.com/questions/364943/gdal-3-0-4-invalid-coordinate-transformation-result
    transformation_srs_to_target = osr.CoordinateTransformation(srs_source, srs_target)

    data_projected = []
    for array in data:
        array_out = np.zeros(shape = array.shape)
        for i in range(len(array)):
            array_out[i,:] = transformation_srs_to_target.TransformPoint(array[i,0],array[i,1])[0:2]
        data_projected.append(array_out)



    Data = {
        'polygons': data_projected,
        'proj_wkt': proj_wkt_target
    }
    return Data

#@jit(forceobj = False)
#def Project_Array(x_in:np.ndarray, y_in:np.ndarray, proj_wkt_source:str, proj_wkt_target:str):
#
#    x_out = np.zeros(shape = x_in.shape)
#    y_out = np.zeros(shape = x_in.shape)
#
#    srs_source = osr.SpatialReference()
#    srs_target = osr.SpatialReference()
#
#    srs_source.ImportFromWkt(proj_wkt_source)
#    srs_target.ImportFromWkt(proj_wkt_target)
#
#    transformation_srs_to_target = osr.CoordinateTransformation(srs_source, srs_target)
#
#    n,m = x_in.shape
#    for i in prange(n):
#        for j in prange(m):
#
#            x_pt = x_in[i,j]
#            y_pt = y_in[i,j]
#            pts_projected = transformation_srs_to_target.TransformPoint(x_pt,y_pt)[0:2]
#
#            x_out[i,j] = pts_projected[0]
#            y_out[i,j] = pts_projected[1]
#
#    Data = {
#        'x_projected': x_out,
#        'y_projected': y_out
#    }
#    return Data
