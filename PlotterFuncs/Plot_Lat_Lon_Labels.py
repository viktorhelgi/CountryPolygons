import matplotlib.pyplot as plt
import numpy as np
from osgeo import osr
from osgeo import gdal

def fmt(x):
    sign = x/abs(x)
    decimals = abs(x)%1

    Minutes = round(decimals*(60/100),3)
    degrees = abs(x)//1
    if sign == -1:
        out = sign*(abs(degrees) + Minutes )
    else:
        out = sign*(degrees + Minutes)
    out = f"{out:.2f}"
    return rf"{out}" if plt.rcParams["text.usetex"] else f"{out}"



def GetLatLon_Labels(x_l, y_l, Location, false_easting, false_northing, proj_target):
    x_min = x_l.min()
    y_min = y_l.min()
    x_max = x_l.max()
    y_max = y_l.max()

    if Location == 'Breidafjordur':
        spacing = 0.5
    elif Location == 'Faxafloi':
        spacing = 0.25
    else:
        #x_spacing = (x_max - x_min)/10
        #y_spacing = (y_max - y_min)/10
        #spacing = (x_spacing + y_spacing)/2
        spacing = 1


    x_min += spacing - x_min%spacing
    y_min += spacing - y_min%spacing
    x_max += spacing - x_max%spacing
    y_max += spacing - y_max%spacing

    Lons = np.arange(x_min, x_max , spacing).astype('float64')
    Lats = np.arange(y_min, y_max , spacing).astype('float64')
    L_x = np.arange(x_min, x_max,   spacing)
    L_y = np.arange(y_min, y_max,   spacing)
    tx = Lons.mean()
    ty = Lats.mean()



    if Location == 'Breidafjordur':
        #ty += spacing/10
        #tx -= spacing/2
        pass
    elif Location == 'Faxafloi':
        ty += - spacing/2
    else:
        ty += ty%spacing - spacing/2

    Loc_x = []
    Loc_y = []

    if proj_target == 'PROJCS["UTM Zone 27, Northern Hemisphere",GEOGCS["unnamed ellipse",DATUM["unknown",SPHEROID["unnamed",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-21],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["Meter",1]]':
        Proj_source = '+proj=longlat +a=6367470 +b=6367470 +no_defs'
        Proj_target = proj_target
        srs_source = osr.SpatialReference()
        srs_target = osr.SpatialReference()
        srs_source.ImportFromProj4(Proj_source)
        srs_target.ImportFromWkt(Proj_target)
    elif proj_target[0] == '+':
        Proj4_source = '+proj=longlat +a=6367470 +b=6367470 +no_defs'
        Proj4_target = proj_target
        srs_source = osr.SpatialReference()
        srs_target = osr.SpatialReference()
        srs_source.ImportFromProj4(Proj4_source)
        srs_target.ImportFromProj4(Proj4_target)
    else:
        Proj4_source = '+proj=longlat +a=6367470 +b=6367470 +no_defs'
        srs_source = osr.SpatialReference()
        srs_target = osr.SpatialReference()
        print()
        srs_source.ImportFromProj4(Proj4_source)
        srs_target.ImportFromWkt(proj_target)
        #srs_source = osr.SpatialReference(Proj4_source)
        #srs_target = osr.SpatialReference(Proj4_target)



    transform = osr.CoordinateTransformation(srs_source, srs_target)


    for lon in Lons:
        geo_pt = transform.TransformPoint(lon, ty)[:2]
        pt = (geo_pt[0], geo_pt[1])
        Loc_x.append(pt)

    for lat in Lats:
        geo_pt = transform.TransformPoint(tx, lat)[:2]
        pt = (geo_pt[0], geo_pt[1])
        Loc_y.append(pt)



    return L_x, L_y, Loc_x, Loc_y
