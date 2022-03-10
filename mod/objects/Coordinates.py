#------------------------------------------------------------------------------
from __future__ import annotations # used for advanced typehinting
#------------------------------------------------------------------------------
import numpy as np
import math
from osgeo import osr
from dataclasses import dataclass, field
from typing import Iterator, List, Generator
from shapely.ops import nearest_points
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.multipoint import MultiPoint
#------------------------------------------------------------------------------
from mod.functions.ShapelyNumpyTransformation import poly_arr_tr
from mod.objects.Projections import (
    Projection, ProjectionRefDefinition, NotDefined
)
#------------------------------------------------------------------------------

@dataclass
class Coordinate:
    x:float
    y:float
    proj:Projection

    def __str__(self):
        return "".join([
            f"Coordinate(\n  x = {round(self.x,4)},",
            f"\n  y = {round(self.y,4)},",
            f"\n  proj = {self.proj.__str_short__()}"
            "\n)"
        ])

    def dist(self, other:Coordinate) -> int:
        """ The distance between two Coordinates """
        return np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

    def project(self, proj_target:Projection) -> Coordinate:
        """ Project coordinate to another projection. """
        if not issubclass(type(proj_target), ProjectionRefDefinition):
            raise TypeError('The Given proj is not of correct type.')
        trfm = osr.CoordinateTransformation(
            self.proj.SpatialReferenceSystem,
            proj_target.SpatialReferenceSystem)
        coord_p:List[float] = trfm.TransformPoint(self.x, self.y)[0:2]
        return Coordinate(x=coord_p[0], y=coord_p[1], proj=proj_target)

    def __isComparable__(self, other:Coordinate) -> bool:
        """ Check if the other Coordinates have the same projection """
        if self.proj!=other.proj:
            return False
        return True

@dataclass(order=True)
class CoordinateCollection:
    """ Hardly used """
    coordinates:List[Coordinate]
    proj:Projection = field(init=False)
    def __post_init__(self):
        if len(self.coordinates) != 0:
            for i in range(0,len(self.coordinates)-1):
                coord1 = self.coordinates[i]
                coord2 = self.coordinates[i+1]
                if coord1.proj != coord2.proj:
                    raise TypeError(
                        "The Projection of all the coordinates is not the same"+
                        f"coord nr={i}\n {coord1}\ncoord nr={i+1}\n {coord2}"
                    )
            self.proj = self.coordinates[0].proj
    def __iter__(self) -> Generator[Coordinate,None,None]:
        yield from self.coordinates
    def __getitem__(self,index):
        return self.coordinates[index]
    @property
    def x(self) -> np.ndarray:
        x_values = []
        for coord in self.coordinates:
            x_values.append(coord.x)
        return np.array(x_values)
    @property
    def y(self) -> np.ndarray:
        y_values = []
        for coord in self.coordinates:
            y_values.append(coord.y)
        return np.array(y_values)

@dataclass
class LineCoord:
    """ A Line represented with two x-y pairs  """
    x1:float
    y1:float
    x2:float
    y2:float
    proj:Projection
    def ToLineString(self)->LineString:
        """ create a shapely linestring [from shapely.geometry import LineString] """
        return LineString([(self.x1,self.y1), (self.x2,self.y2)])
    @property
    def a(self):
        """ The slope of the line [ y = a*x + k ] """
        return (self.y2-self.y1)/(self.x2-self.x1)
    @property
    def k(self):
        """ The intercept of the line with the y-axis """
        return self.y1 - self.a*self.x1
    @property
    def inv_a(self):
        """ The slope of the line which is perpendicular to the object-line """
        return -1/self.a
    def inv_k(self, coord:Coordinate):
        """ The intercept of the line which is perpendicular to the object-line """
        return coord.y-self.inv_a*coord.x
    def closest_coord_on_line(self, coord:Coordinate) -> List[float]:
        """ The xy coordinate located on the object-line closest to the input coordinate """
        a1 = self.a
        a2 = -1/a1
        k1 = self.k
        k2 = self.inv_k(coord)
        xf = (k2-k1)/(a1-a2)
        yf = a2*xf+k2
        return [xf, yf]
    #def dist(self, coord:Coordinate):
    #    return (self.a*coord.x - coord.y + self.k)/math.sqrt(self.a**2 + 1)
    def minimum_distance(self, coord:Point) -> float:
        """ The minimum distance from the coordinate to the object-line """
        line = self.ToLineString()
        return coord.distance(line)
    def point_on_line(self, coord:Point)->np.ndarray:
        """
        The location on the object-line providing the minimum distance from the
        coordinate to the object-line
        """
        line = self.ToLineString()
        return np.array(nearest_points(coord, line)[1].coords).ravel()
    def contains(self, coord:Coordinate) -> bool:
        """ Not used """
        print('WARNING: the class function LineCoord.contains was used')

        x_min = np.min([self.x1,self.x2])
        x_max = np.max([self.x1,self.x2])
        y_min = np.min([self.y1,self.y2])
        y_max = np.max([self.y1,self.y2])
        return (
            x_min <= coord.x <= x_max
        )*(
            y_min <= coord.y <= y_max
        )



@dataclass
class LineCoords:
    x:np.ndarray
    y:np.ndarray
    proj:Projection
    @property
    def matrix(self) -> np.ndarray:
        """ return the x and y arrays as a combined matrix of size [N,2] """
        return np.vstack([self.x, self.y]).T
    def project(self, proj_target:Projection) -> LineCoords:
        """
        Project Coordinate to another projection.
        -----------------------------------------------------------------------
        This function copys the coordinate, it doesn't change the original one.
        """
        if not issubclass(type(proj_target), ProjectionRefDefinition):
            raise TypeError('The Given proj is not of correct type.')

        n = self.x.shape[0]
        x_tg = np.zeros(shape=(n))
        y_tg = np.zeros(shape=(n))

        trfm = osr.CoordinateTransformation(
            self.proj.SpatialReferenceSystem,
            proj_target.SpatialReferenceSystem
        )

        for pt_i in range(n):
            pts_projected = trfm.TransformPoint(
                self.x[pt_i], self.y[pt_i])[0:2]
            x_tg[pt_i] = pts_projected[0]
            y_tg[pt_i] = pts_projected[1]
        return LineCoords(x=x_tg, y=y_tg, proj=proj_target)

@dataclass
class PolyCoords(LineCoords):
    """
    x and y are one dimensional arrays
    """
    x:np.ndarray
    y:np.ndarray
    proj:Projection
    def __len__(self):
        return self.x.shape[0]
    def line_iterator(self) -> Iterator[LineCoord]:
        n=self.x.shape[0]
        for i in range(0,n-1):
            x1 = self.x[i]
            y1 = self.y[i]
            x2 = self.x[i+1]
            y2 = self.y[i+1]
            yield LineCoord(x1,y1,x2,y2,self.proj)

    @property
    def matrix(self):
        """ return the x and y arrays as a combined matrix of size [N,2] """
        return np.vstack([self.x, self.y]).T
    def project(self, proj_target:Projection) -> PolyCoords:
        """
        Project Coordinate to another projection.
        -----------------------------------------------------------------------
        This function copys the coordinate, it doesn't change the original one.
        """
        if not issubclass(type(proj_target), ProjectionRefDefinition):
            raise TypeError('The Given proj is not of correct type.')

        n = self.x.shape[0]
        x_tg = np.zeros(shape=(n))
        y_tg = np.zeros(shape=(n))

        trfm = osr.CoordinateTransformation(
            self.proj.SpatialReferenceSystem,
            proj_target.SpatialReferenceSystem
        )

        for pt_i in range(n):
            pts_projected = trfm.TransformPoint(
                self.x[pt_i], self.y[pt_i])[0:2]
            x_tg[pt_i] = pts_projected[0]
            y_tg[pt_i] = pts_projected[1]

        return PolyCoords(x=x_tg, y=y_tg, proj=proj_target)
    def GetLineString(self, add_endpoint=True) -> LineString:
        if add_endpoint:
            points = [[x,y] for x,y in zip(self.x, self.y)]
            points.append([self.x[0], self.y[0]])
            return LineString(points)
        return LineString([[x,y] for x,y in zip(self.x, self.y)])
    def GetShapelyPolygon(self) -> Polygon:
        return Polygon([[x,y] for x,y in zip(self.x, self.y)])

    def simplify(self, tolerance=100, preserve_topology=False):
        polygon = self.GetShapelyPolygon()
        polygon = polygon.simplify(tolerance, preserve_topology=preserve_topology)
        array = np.array(polygon.boundary.coords)
        return PolyCoords(x=array[:,0], y=array[:,1], proj=self.proj)

    def offset_polygon(self, dist, side='left', tolerance=100, preserve_topology=False) -> PolyCoords:
        """
        This function creates a new polygon which is either larger or smaller
        than the original polygon, by a distance of 'dist'
        -----------------------------------------------------------------------
        May need to be fixed. was originally only created for a single map.
        Trouble occurred when attempting to close the larger-sized polygon.
        """
        polygon = self.GetShapelyPolygon()
        polygon = polygon.simplify(tolerance,preserve_topology)
        array = np.array(polygon.boundary.coords)
        array = np.concatenate([array, array[0:3,:].reshape([3,2])], axis=0)
        line = LineString([[x,y] for x,y in zip(array[:,0], array[:,1])])
        line_off = line.parallel_offset(dist, mitre_limit=5 , join_style=2, side=side)
        array = np.array(line_off.coords)
        array = np.concatenate([array, array[0,:].reshape([1,2])], axis=0)
        return PolyCoords(x=array[:,0], y=array[:,1], proj=self.proj)

@dataclass
class GridCoords:
    """
    Dataclass containing two arrays/matrices 'x' and 'y' which represents locations
    "corresponding" to the given projection 'proj'
    """
    x:np.ndarray
    y:np.ndarray
    proj:Projection

    def project(self, proj_target:Projection) -> GridCoords:
        """
        Project Coordinate to another projection.
        -----------------------------------------------------------------------
        This function copys the coordinate, it doesn't change the original one.
        """
        if not issubclass(type(proj_target), ProjectionRefDefinition):
            raise TypeError('The Given proj is not of correct type.')

        n,m = self.x.shape
        x_tg = np.zeros(shape=(n,m))
        y_tg = np.zeros(shape=(n,m))

        trfm = osr.CoordinateTransformation(
            self.proj.SpatialReferenceSystem,
            proj_target.SpatialReferenceSystem
        )

        for i in range(n):
            for j in range(m):
                pts_projected = trfm.TransformPoint(
                    self.x[i,j], self.y[i,j])[0:2]
                x_tg[i,j] = pts_projected[0]
                y_tg[i,j] = pts_projected[1]

        return GridCoords(x=x_tg, y=y_tg, proj=proj_target)

    def est_area(self) -> float:
        n,m = self.x.shape

        c1 = (self.x[0,0], self.y[0,0])
        c2 = (self.x[n-1,0], self.y[n-1,0])
        c3 = (self.x[0,m-1], self.y[0,m-1])
        c4 = (self.x[n-1,m-1], self.y[n-1,m-1])

        # triangles
        tr1 = np.array([
            [c1[0], c1[1], 1],
            [c2[0], c2[1], 1],
            [c3[0], c3[1], 1]])
        tr2 = np.array([
            [c2[0], c2[1], 1],
            [c3[0], c3[1], 1],
            [c4[0], c4[1], 1]])
        return abs(0.5*np.linalg.det(tr1) +  0.5*np.linalg.det(tr2))
    def est_density(self) -> float:
        area = self.est_area()
        n,m = self.x.shape
        return n*m/area
    def est_spacing(self) -> float:
        n,m = self.x.shape

        c1 = (self.x[0,0], self.y[0,0])
        c2 = (self.x[n-1,0], self.y[n-1,0])
        c3 = (self.x[0,m-1], self.y[0,m-1])
        c4 = (self.x[n-1,m-1], self.y[n-1,m-1])

        d1 = np.sqrt((c1[0] - c2[0])**2 + (c1[1]-c2[1])**2)/n
        d2 = np.sqrt((c2[0] - c4[0])**2 + (c2[1]-c4[1])**2)/m
        d3 = np.sqrt((c1[0] - c3[0])**2 + (c1[1]-c3[1])**2)/m
        d4 = np.sqrt((c3[0] - c4[0])**2 + (c3[1]-c4[1])**2)/n
        return (d1+d2+d3+d4)/4

    def convexhull(self) -> PolyCoords:
        points = []
        x,y = self.x, self.y
        n,m = x.shape
        x_bound = np.concatenate([x[:,0], x[:,m-1], x[0,:], x[n-1,:]])
        y_bound = np.concatenate([y[:,0], y[:,m-1], y[0,:], y[n-1,:]])

        for i,j in zip(x_bound,y_bound):
            points.append((i,j))

        multi_point = MultiPoint(points)
        polygon_hull = multi_point.convex_hull
        array:np.ndarray = poly_arr_tr(polygon_hull)
        if type(array)!=np.ndarray:
            raise TypeError('This is supposed to return an np.ndarray on default')
        return PolyCoords(
            x= array[:,0],
            y = array[:,1],
            proj = self.proj
        )





@dataclass
class PolyCoordsCollection:
    polys:List[PolyCoords]
    proj:Projection

    def __iter__(self) -> Generator[PolyCoords,None,None]:
        yield from self.polys
    def __getitem__(self,index):
        return self.polys[index]
    def project(self, proj_target:Projection) -> PolyCoordsCollection:
        poly_p = []
        for poly in self.polys:
            poly_p.append(poly.project(proj_target))
        return PolyCoordsCollection(
            polys = poly_p, proj=proj_target
        )
    def GetShapelyPolygons(self) -> List[Polygon]:
        return [poly.GetShapelyPolygon() for poly in self.polys]


