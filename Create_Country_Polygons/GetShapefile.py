from osgeo import osr
import os
import fiona
import numpy as np
import pickle


from Objects.Coordinates import PolyCoords, PolyCoordsCollection
from Objects.Projections import WorldGeodeticSystem


files = {
    'denmark': './data/geoBoundaries/denmark/geoBoundaries-DNK-ADM1.shp',
    'norway': './data/geoBoundaries/norway/geoBoundaries-NOR-ADM0.shp'
}
def Get_Shapefile(file:str, seperate_polygons = True):

    # If polygons are seperated, then a list with all the polygons is returned in a list
    # else, a string is returned
    with fiona.open(file) as Shapefile:
        polys = []
        count = 0
        proj = Shapefile.crs_wkt
        for raster in Shapefile:
            raster_geometry = raster['geometry']
            if raster_geometry['type'] == 'Polygon' and len(raster_geometry['coordinates']) == 1:
                Polygon = raster_geometry['coordinates']
                array = np.array(Polygon)
                maxx = max(array.shape)
                poly = array.reshape(maxx, 2)
                polys.append(poly)

            elif raster_geometry['type'] == 'MultiPolygon':
                Multipolygon = raster_geometry['coordinates']
                for Polygon in Multipolygon:
                    for coords in Polygon:
                        poly = np.array(coords)
                        polys.append(poly)
                        assert len(poly.shape) == 2
                        assert poly.shape[0] > poly.shape[1]

            else:
                Polygon = raster_geometry['coordinates']
                for coords in Polygon:
                    poly = np.array(coords)
                    polys.append(poly)

            count += 1

    return {'polygons':polys, 'proj_wkt':proj}


if __name__ == '__main__':
    out_no = Get_Shapefile(file=files['norway'])
    out_dk = Get_Shapefile(file=files['denmark'])

    output_file_dk = 'data/pickles/denmark_PolyCoords.p'
    output_file_no = 'data/pickles/norway_PolyCoords.p'
    with open(output_file_dk, 'wb') as handle:
        pickle.dump(out_dk, handle)
    with open(output_file_no, 'wb') as handle:
        pickle.dump(out_no, handle)








