from __future__ import annotations


from osgeo import osr
from dataclasses import dataclass
#from functions.Objects import Forecast
import datetime
from dataclasses import dataclass, field

from typing import Union, List, Optional, Protocol, overload, Generator

import numpy as np

from ProjectionObjects import Projection, WorldGeodeticSystem, ProjectionRefDefinition, UnknownProjection

from shapely.geometry.polygon import Polygon
from shapely.geometry.multipoint import MultiPoint

#------------------------------------------------------------------------------


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

    def dist(self,other) -> int:
        return np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

    def project(self, proj_target:Projection) -> Coordinate:
        """
        Project Coordinate to another projection.
        -----------------------------------------------------------------------
        This function copys the coordinate, it doesn't change the original one.
        """
        if not issubclass(type(proj_target), ProjectionRefDefinition):
            raise TypeError('The Given proj is not of correct type.')
        # Create the Coordinate Transformation object
        trfm = osr.CoordinateTransformation(
            self.proj.SpatialReferenceSystem,
            proj_target.SpatialReferenceSystem
        )
        # Project the coordinates.
        coord_p:List[float] = trfm.TransformPoint(self.y, self.x)[0:2]
        return Coordinate(x=coord_p[0], y=coord_p[1], proj=proj_target)

    def __isComparable__(self, other:Coordinate) -> bool:
        """ Check if the other Coordinate has the same projection """
        if self.proj!=other.proj:
            return False
        return True

@dataclass(order=True)
class CoordinateCollection:
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
class PolyCoords:
    """
    x and y are one dimensional arrays
    """
    x:np.ndarray
    y:np.ndarray
    proj:Projection

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
    def GetShapelyPolygon(self) -> Polygon:
        return Polygon([[x,y] for x,y in zip(self.x, self.y)])


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

        for i,j in zip(
                x_bound,
                y_bound):
            points.append((i,j))

        multi_point = MultiPoint(points)
        polygon_hull = multi_point.convex_hull
        raise Exception()
        #array:np.ndarray = poly_arr_tr(polygon_hull)
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


