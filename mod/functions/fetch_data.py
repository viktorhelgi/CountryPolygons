#------------------------------------------------------------------------------
import pickle
from typing import TypedDict, List
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from numpy import ndarray
#------------------------------------------------------------------------------
from mod.objects.Coordinates import PolyCoords, PolyCoordsCollection
from mod.objects.Projections import WorldGeodeticSystem
#------------------------------------------------------------------------------
import mod.global_data as gd
#------------------------------------------------------------------------------

class __CountryData__(TypedDict):
    boundaries:MultiPolygon
    polygons:List[ndarray]
    main_land:Polygon

def get_country_polygons() -> PolyCoordsCollection:
    """ ...
    gd.polygons_dir:str
        location of the folder where the pickle-files are which have the
        polygons for the countries
    gd.countries:List[str]
        A list containing the countries which will be used.
    ---------------------------------------------------------------------------
    These polygons are used to change the values of the forecasts which
    coordinates are located inside countries (that is, on land)
    """
    polygons_all = []
    for country in gd.countries:
        with open(f'{gd.polygons_dir}/{country}.p', '+rb') as handle:
            country_data:__CountryData__ = pickle.load(handle)
            polygons = country_data['polygons']
        for poly in polygons:
            polygons_all.append(
                PolyCoords(
                    x=poly[:,0],
                    y=poly[:,1],
                    proj=WorldGeodeticSystem()
                )
            )
    return PolyCoordsCollection(
        polys=polygons_all,
        proj=WorldGeodeticSystem()
    )
