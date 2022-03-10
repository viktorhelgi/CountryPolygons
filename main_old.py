# conda environment: shapefiles
import pickle
import fiona
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Literal, Union, Tuple, TypedDict, Iterator
from dataclasses import dataclass

from osgeo import osr

from shapely.geometry.polygon import Polygon

from Objects.Coordinates import PolyCoords, PolyCoordsCollection
from Objects.Projections import WorldGeodeticSystem

files = {
    'denmark': 'data/geoBoundaries/denmark/geoBoundaries-DNK-ADM1.shp',
    'norway':  'data/geoBoundaries/norway/geoBoundaries-NOR-ADM0.shp'
}


class Geom(TypedDict):
    coordinates:List[List[List[Tuple[float,float]]]]
    type:Literal['MultiPolygon']

class Record(TypedDict):
    type:str
    id: str
    properties: OrderedDict
    geometry:Geom

fiona.collection
@dataclass
class GeoBoundaries:
    file:str
    def print_info(self)->None:
        with fiona.open(self.file) as shp:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(shp.crs_wkt)
            print(srs.ExportToPrettyWkt())
    def generator(
            self,
            out_type:Literal['Polygon', 'PolyCoords']='PolyCoords'
            ) -> Iterator[Union[Polygon,PolyCoords]]:
        with fiona.open(self.file) as shp:
            rec:Record
            #proj = shp.crs_wkt()
            for rec in shp:
                if rec['geometry']['type'] == 'MultiPolygon':
                    for polygon in rec['geometry']['coordinates']:
                        if out_type=='Polygon':
                            if len(polygon)==1:
                                polygon = polygon[0]
                            yield Polygon([[x,y] for x,y in polygon])
                        elif out_type=='PolyCoords':
                            try:
                                print(len(polygon), len(polygon[0]), len(polygon[1]))
                            except:
                                pass
                            array = np.array(polygon).reshape([-1,2])
                            print(array.shape)
                            yield PolyCoords(
                                x = array[:,0],
                                y = array[:,1],
                                proj = WorldGeodeticSystem()
                            )
            print(shp.crs_wkt)

def main(file) -> PolyCoordsCollection:
    country = GeoBoundaries(file)
    polys = []
    for poly in country.generator():
        polys.append(poly)
    return PolyCoordsCollection(polys, proj=WorldGeodeticSystem())






if __name__ == '__main__':

    out_no = main(file=files['norway'])
    print(type(out_no))

    import sys
    sys.exit()
    out_dk = main(file=files['denmark'])

    output_file_dk = 'data/pickles/denmark_PolyCoords.p'
    output_file_no = 'data/pickles/norway_PolyCoords.p'
    with open(output_file_dk, 'wb') as handle:
        pickle.dump(out_dk, handle)
    with open(output_file_no, 'wb') as handle:
        pickle.dump(out_no, handle)

















