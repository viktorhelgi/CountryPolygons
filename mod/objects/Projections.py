#------------------------------------------------------------------------------
from __future__ import annotations
#------------------------------------------------------------------------------
from dataclasses import dataclass, field
from osgeo import osr
from typing import Optional, Union, Literal, List
from shapely.impl import ImplementationError
#------------------------------------------------------------------------------
# These classes are used to represent different kinds of
# - Projections, Spatial Reference Systems, and/or Coordination Systems
@dataclass(eq=True)
class ProjectionRefDefinition:
    EPSG:Optional[int] = field(default=None, init=False)
    wkt:Optional[str] = field(default=None, init=False)
    proj4:Optional[str] = field(default=None, init=False)
    def __str_short__(self) -> str:
        if self.proj4!=None:
            return self.proj4
        elif self.EPSG!=None:
            return str(self.EPSG)
        elif self.wkt!=None:
            return self.wkt[0:25]+'...'
        return 'None'
    def __post_init__(self) -> None: ...
    def __eq__(self, other:ProjectionRefDefinition) -> bool:
        if self.EPSG!=other.EPSG or self.wkt!=other.wkt or self.proj4!=other.proj4:
            return False
        return True
    def SetWKT(self, wkt:str) -> None:
        if type(wkt) != str:
            raise TypeError(type(wkt))
        self.wkt = wkt
    def SetProj4(self, proj4:str) -> None:
        if type(proj4) != str:
            raise TypeError(type(proj4))
        self.proj4 = proj4
    def SetEPSG(self, EPSG:int) -> None:
        if type(EPSG) != int:
            raise TypeError(type(EPSG))
        self.EPSG = EPSG
    @property
    def SpatialReferenceSystem(self) -> osr.SpatialReference:
        """
        Create an SpatialReference
        -----------------------------------------------------------------------
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        This is the reason why we use this
        - https://gis.stackexchange.com/questions/364943/gdal-3-0-4-invalid-coordinate-transformation-result
        - https://github.com/OSGeo/gdal/issues/1546
        """
        srs = osr.SpatialReference()
        if self.wkt != None:
            srs.ImportFromWkt(self.wkt)
        elif self.proj4 != None:
            srs.ImportFromProj4(self.proj4)
        elif self.EPSG != None:
            srs.ImportFromEPSG(self.EPSG)
        else:
            raise Exception()
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) #https://gis.stackexchange.com/questions/364943/gdal-3-0-4-invalid-coordinate-transformation-result
        return srs
    def GetProjectionRef(self) -> Union[str,int]:
        if self.wkt != None:
            return self.wkt
        elif self.proj4 != None:
            return self.proj4
        elif self.EPSG != None:
            return self.EPSG
        raise Exception("All projectionRefs are None")
@dataclass
class UnknownProjection(ProjectionRefDefinition):
    type:Literal['wkt','proj4','EPSG', 'undefined']
    ProjectionRef:Union[str,int]
    def __post_init__(self):
        self.wkt = None
        self.proj4 = None
        self.EPSG = None
        setattr(self, self.type, self.ProjectionRef)
#------------------------------------------------------------------------------
@dataclass
class OpenStreetMaps(ProjectionRefDefinition):
    EPSG:int = field(init=False)
    def __post_init__(self):
        self.EPSG = 3857

@dataclass
class WorldGeodeticSystem(ProjectionRefDefinition):
    """Latitude and Longitude. Not really a projection"""
    EPSG:int = field(init=False)
    def __post_init__(self):
        self.EPSG = 4326

@dataclass
class UniversalTransverseMercator(ProjectionRefDefinition):
    zone: int = 27
    central_meridian:int = field(default=-21, metadata={'units':'degrees'})
    latitude_of_origin:int = field(default=0, metadata={'units':'degrees'})
    k:float = field(default=1, metadata={'description':'scale_factor'})
    wkt:str = field(init=False)
    def __post_init__(self):
        self.wkt  = ''.join([
            f'PROJCS["UTM Zone {self.zone}, Northern Hemisphere",GEOGCS[',
            '"WGS 84",DATUM["unknown",SPHEROID["WGS84",6378137',
            ',298.257223563]],PRIMEM["Greenwich",0],UNIT["degr',
            'ee",0.0174532925199433]],PROJECTION["Transverse_M',
            f'ercator"],PARAMETER["latitude_of_origin",{self.latitude_of_origin}],PARAM',
            f'ETER["central_meridian",{self.central_meridian}],PARAMETER["scale_fac',
            f'tor",{self.k}],PARAMETER["false_easting",500000],PA',
            'RAMETER["false_northing",0],UNIT["Meter",1]]'])

@dataclass
class ObliqueMercator(ProjectionRefDefinition):
    false_easting:int = 0
    false_northing:int = 0
    lon_1:float = field(default=0, metadata={'units':'degrees'})
    lat_1:float = field(default=0, metadata={'units':'degrees'})
    lon_2:float = field(default=0, metadata={'units':'degrees'})
    lat_2:float = field(default=0, metadata={'units':'degrees'})
    proj4:str = field(init=False)
    x_1:float = field(init=False, metadata={'units':'meters'})
    y_1:float = field(init=False, metadata={'units':'meters'})
    x_2:float = field(init=False, metadata={'units':'meters'})
    y_2:float = field(init=False, metadata={'units':'meters'})

    def __init_method1__(self):
        """
        Create the projection given to coordinates (the coordinates are defined
        in latitudes and longitudes). False easting and false northing is also
        used in the definition.
        Afterwards, the two coordinates are projected onto the projection.
        """
        self.proj4 = ''.join([
            f'+proj=omerc ',
            f'+lon_1={self.lon_1} ',
            f'+lat_1={self.lat_1} ',
            f'+lon_2={self.lon_2} ',
            f'+lat_2={self.lat_2} ',
            f'+x_0={self.false_easting} ',
            f'+y_0={self.false_northing}'
        ])
        # Create the Coordinate Transformation object
        trfm = osr.CoordinateTransformation(
            WorldGeodeticSystem().SpatialReferenceSystem,
            self.SpatialReferenceSystem)
        # Project the coordinates.
        coord1_m:List[float] = trfm.TransformPoint(self.lon_1,self.lat_1)[0:2]
        coord2_m = trfm.TransformPoint(self.lon_2,self.lat_2)[0:2]
        self.x_1,self.y_1 = coord1_m
        self.x_2,self.y_2 = coord2_m
    def __init_method2__(self):
        pass
    def __init_method3__(self):
        """
        The projection can also be defined given an angle instead of two
        coordinates.
        ----------------------------------------------------------------
        Hasn't been implemented. Further info:
            https://proj.org/operations/projections/omerc.html
        """
        raise NotImplementedError("This function should not have been called")
    def __post_init__(self):
        if self.lon_1!=0 or self.lat_1!=0 or self.lon_2!=0 or self.lat_2!=0:
            self.__init_method1__()
        else:
            pass
            #self.__init_method2__()
    def __str__(self):
        return self.proj4
    def __str_short__(self) -> str:
        if self.proj4!=None:
            return self.proj4
        elif self.EPSG!=None:
            return str(self.EPSG)
        elif self.wkt!=None:
            return self.wkt[0:25]+'...'
        return 'None'
    def __repr__(self):
        return ''.join([
            f'ObliqueMercator(\n',
            f'  proj4={self.proj4}'])

class NotDefined(ProjectionRefDefinition):
    def SpatialReference(self):
        raise ImplementationError('Projection is notdefined')
    def GetProjectionRef(self):
        raise ImplementationError('Projection is notdefined')
#------------------------------------------------------------------------------
# Used for typehinting
Projection = Union[
    UniversalTransverseMercator,
    ObliqueMercator,
    OpenStreetMaps,
    WorldGeodeticSystem,
    UnknownProjection,
    NotDefined
]
#------------------------------------------------------------------------------
def __example__():
    """ How to check if class instance is a projection """
    obj=WorldGeodeticSystem()
    if issubclass(type(obj), ProjectionRefDefinition):
        print('works')
    if isinstance(obj, ProjectionRefDefinition):
        print('works')

if __name__ == '__main__':
    __example__()


