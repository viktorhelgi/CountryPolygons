import numpy as np
from osgeo import osr

def Project_Points(data, proj_source, proj_target):
    
    srs_source = osr.SpatialReference()
    srs_target = osr.SpatialReference()
    
    if type(proj_source) != int:
        proj_source = osr.GetUserInputAsWKT(proj_source)
        srs_source.ImportFromWkt(proj_source)
    else:
        srs_source.ImportFromEPSG(proj_source)

    if type(proj_target) != int:
        proj_target = osr.GetUserInputAsWKT(proj_target)
        srs_target.ImportFromWkt(proj_target)
    else:
        srs_target.ImportFromEPSG(proj_target)

    transformation_srs_to_target = osr.CoordinateTransformation(srs_source, srs_target)

    data_projected = []
    if type(data) == list:
        for array in data:
            array_out = np.zeros(shape = array.shape)
            for i in range(len(array)):
                array_out[i,:] = transformation_srs_to_target.TransformPoint(array[i,0],array[i,1])[0:2]
            data_projected.append(array_out)
        Data = {
            'polygons': data_projected,
            'proj_wkt': proj_target
        }
    else: # if array
        array = data
        array_out = np.zeros(shape = array.shape)
        for i in range(len(array)):
            array_out[i,:] = transformation_srs_to_target.TransformPoint(array[i,0],array[i,1])[0:2]
        Data = {
            'array': array_out,
            'proj_wkt': proj_target
        }
        
    return Data
