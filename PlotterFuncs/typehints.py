from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from osgeo.gdal import Band, gdalconst
from osgeo.ogr import Layer
from numpy import ndarray
from dataclasses import field
from typing import List, Tuple, Any, Callable, Union, Protocol, TypedDict

#C = TypeVar('C', bound=Callable, covariant=True)
#class __ReadAsArray(Protocol[C]):
#    def __call__(
#        self, xoff=0, yoff=0, xsize=None, ysize=None, buf_obj=None,
#        buf_xsize=None, buf_ysize=None, buf_type=None,
#        resample_alg=gdalconst.GRIORA_NearestNeighbour, callback=None,
#        callback_data=None, interleave='band', band_list=None
#    ) -> np.ndarray: ...


class __CP_boundaries__(TypedDict):
    polygons:List[ndarray]     # size of np.ndarrays -> Nx2
    proj_wkt:int
class Country_Polygon(TypedDict):
    """
    This is the input in the pickle files containing the country boundaries
    """
    Polygons:MultiPolygon
    Main_Land:Polygon
    proj_wkt:Union[str,int]
    boundaries:__CP_boundaries__


class Coords(TypedDict):
    """
    Dictionary containing two matrices of size NxM containing the coordinates
    of each point.
    """
    x:ndarray
    y:ndarray

#@dataclass()
class GDAL_DataObj(Protocol):
    """Type hints for the class Dataset in the gdal module (gdal.Dataset)"""
    AbortSQL:Any = None
    AddBand:Any = None
    AddFieldDomain:Any = None
    AdviseRead:Any = None
    BeginAsyncReader:Any = None
    BuildOverviews:Any = None
    ClearStatistics:Any = None
    CommitTransaction:Any = None
    CopyLayer:Any = None
    CreateLayer:Any = None
    CreateMaskBand:Any = None
    DeleteLayer:Any = None
    EndAsyncReader:Any = None
    ExecuteSQL:Any = None
    FlushCache:Any = None
    GetDriver:Any = None
    GetFieldDomain:Any = None
    GetFileList:Any = None
    GetGCPCount:Any = None
    GetGCPProjection:Any = None
    GetGCPSpatialRef:Any = None
    GetGCPs:Any = None
    GetMetadataDomainList:Any = None
    GetMetadataItem:Any = None
    GetMetadata_List:Any = None
    GetNextFeature:Any = None
    GetProjection:Any = None
    GetProjectionRef:Any = None
    GetRootGroup:Any = None
    GetSpatialRef:Any = None
    GetStyleTable:Any = None
    GetSubDatasets:Callable[...,List[tuple[str,str]]] = field(default=list)
    GetTiledVirtualMem:Any = None
    GetTiledVirtualMemArray:Any = None
    GetVirtualMem:Any = None
    GetVirtualMemArray:Any = None
    IsLayerPrivate:Any = None
    RasterCount:Any = None
    RasterXSize:Any = None
    RasterYSize:Any = None
    #ReadAsArray:Callable[..., np.ndarray] = np.array
    #ReadAsArray:__ReadAsArray  = Any
    #ReadAsArray:_ReadAsArray = np.ndarray
    ReadRaster:Any = None
    ReadRaster1:Any = None
    ReleaseResultSet:Any = None
    ResetReading:Any = None
    RollbackTransaction:Any = None
    SetDescription:Any = None
    SetGCPs:Any = None
    SetGeoTransform:Any = None
    SetMetadata:Any = None
    SetMetadataItem:Any = None
    SetProjection:Any = None
    SetSpatialRef:Any = None
    SetStyleTable:Any = None
    StartTransaction:Any = None
    TestCapability:Any = None
    WriteArray:Any = None
    WriteRaster:Any = None
    def ReadAsArray(
        self, xoff:int=0, yoff:int=0, xsize:int=None, ysize:int=None, buf_obj=None,
        buf_xsize=None, buf_ysize=None, buf_type=None,
        resample_alg=gdalconst.GRIORA_NearestNeighbour, callback=None,
        callback_data=None, interleave='band', band_list=None
    ) -> ndarray: ...
    def GetLayerByIndex(self, index:int) -> Layer: ...
    def GetLayerByName(self, layer_name:str) -> Layer: ...
    def GetLayer(self, iLayer:Union[str,int]=0 ) -> Layer: ...
    def GetGeoTransform(self) -> Tuple[float,float,float,float,float,float]: ...
    def GetDescription(self) -> str: ...
    def GetLayerCount(self) -> int: ...
    def GetRasterBand(self,nBand:int) -> Band: ...
    def GetMetadata(self) -> dict: ...
    def GetMetadata_Dict(self) -> dict: ...


