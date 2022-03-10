#------------------------------------------------------------------------------
from __future__ import annotations
#------------------------------------------------------------------------------
import numpy as np
import math
import datetime
import os
import pickle
import copy
from typing import List, Union, Tuple, Optional, Iterator, Generator, Literal
from dataclasses import dataclass
from shapely.impl import ImplementationError
from shapely.strtree import STRtree
from shapely.geometry import Point
from scipy.interpolate import (
    CloughTocher2DInterpolator, NearestNDInterpolator
)
from matplotlib.path import Path
#------------------------------------------------------------------------------
import mod.global_data as gd
import mod.functions.fetch_data as fd
from mod.objects.Coordinates import (
    Coordinate, GridCoords, PolyCoords, PolyCoordsCollection)
from mod.objects.Projections import NotDefined, UnknownProjection, Projection
#------------------------------------------------------------------------------
class __ForecastInputError__(Exception):
    """
    The three matrices, x,y, and z
    """
    pass
class __ForecastCombinationError__(Exception):
    pass
#------------------------------------------------------------------------------

@dataclass(order=True)
class PointForecast(Coordinate):
    z:float
    date:datetime.datetime
    element:str = 'UNDEFINED'
    # Default Valued Variables
    source:Union[str,List[str]] = 'unspecified'



@dataclass(order=True)
class Forecast(GridCoords):
    """
    This dataclass contains a forecast which has been obtained from a raster
    file (i.e. nc-file and grib-file)
    ---------------------------------------------------------------------------
    All matrices *x*, *y* and *z* are of the same size.
    priority: Used to rank GribArea-objects based on their resolution
    proj: Not specified when GribDataHolder is used
    element: Not specified when GribDataHolder is used
    ---------------------------------------------------------------------------
    As of now, we are not using priority. However, this attribute can be used
    to priorities which forecasts shall be used where.
    """
    # Non-default value variables
    z:np.ndarray
    date:datetime.datetime
    element:str = 'UNDEFINED'
    # Default Valued Variables
    source:Union[str,List[str]] = 'unspecified'
    """ ... add later
    spacing:float = field(init=False, metadata={'units':'degrees'})
    priority:???
    """

    def point_iterator(self) -> Iterator[PointForecast]:
        for x,y,z in zip(self.x.flatten(),self.y.flatten(),self.z.flatten()):
            yield PointForecast(
                x, y, self.proj, z, self.date, self.element, self.source)

    def __str__(self) -> str:
        output = "".join([
            f'Forecast(\n  Matrices: x, y, z [{self.x.shape}]',
            f'\n  Projection: {type(self.proj).__name__}',
            f'\n  Date: {self.date}',
            f'\n  Element: {self.element}',
            f'\n  Source: {self.source})'])
        return output

    def __isComparable__(self, other:Forecast) -> bool:
        if self.date!=other.date or self.proj!=other.proj:
            return False
        return True

    def __add__(self, other:Forecast) -> Forecast:
        """...
        Return a New Forecast
        """
        if not  self.__isComparable__(other):
            raise __ForecastCombinationError__(
                "\nThe Forecasts can't be combined, either they have different "+
                f"dates or projecctions. \n{self.proj = } \n{other.proj = }"+
                f"\n{self.date = }\n{other.date = }")
        x_new = np.concatenate([self.x.flatten(), other.x.flatten()]).T
        y_new = np.concatenate([self.y.flatten(), other.y.flatten()]).T
        z_new = np.concatenate([self.z.flatten(), other.z.flatten()]).T
        def __combine_sources__(
                source1:str|List[str], source2:str|List[str]) -> List[str]:
            output = []
            output.extend(source1)
            output.extend(source2)
            return output
        sources = __combine_sources__(self.source, other.source)

        return Forecast(
            x=x_new, y=y_new, z=z_new,
            date=self.date, proj=self.proj,
            element=self.element, source=sources)
    @property
    def grid(self) -> GridCoords:
        return GridCoords(self.x, self.y, self.proj)

    def interpolate_CloughTocher(self, grid:GridCoords) -> Forecast:
        """
        Use Clough Tocher interpolation to calculate the z-values corresponding
        to the locations in the grid.
        """

        def __fix_outofbounds__(
            z_bef:np.ndarray,
            z_aft:np.ndarray
        ) -> np.ndarray:
            """
            The CloughTocher2DInterpolator is biased near the shoreline, this is
            because there are no datapoints on land. However, the model assumes
            those datapoints will be negative. Thus, the program might return
            Negative values
            """
            z_min = np.nanmin(z_bef)
            z_max = np.nanmax(z_bef)
            z_aft[z_aft<=z_min] = z_min
            z_aft[z_aft>=z_max] = z_max
            return z_aft


        if grid.x.shape[0]*grid.x.shape[0] <= 1:
            raise Exception(f"\nUnable to interpolate. \n{grid.x.shape=}\n{grid.y.shape=}")
        if self.x.shape!=self.y.shape or self.y.shape!=self.z.shape:
            raise Exception(f"\n{self.x.shape=}\n{self.y.shape=}\n{self.z.shape=}")

        x = self.x[np.isnan(self.z)==False].reshape([-1,1])
        y = self.y[np.isnan(self.z)==False].reshape([-1,1])
        z = self.z[np.isnan(self.z)==False].reshape([-1,1])
        coords = np.concatenate([x, y], axis=1)

        model = CloughTocher2DInterpolator(
            points=coords,
            tol=1e-13,
            maxiter=100,
            values=z,
            rescale=True
        )
        z_calc = model(grid.x, grid.y).squeeze()
        z_out = __fix_outofbounds__(z_bef=self.z, z_aft=z_calc)

        return Forecast(
            x=grid.x, y=grid.y, z=z_out, date=self.date, proj=self.proj,
            element=self.element, source=self.source)

    @dataclass
    class Subset:
        """
        This subclass of the class Forecast was created to manage the
        combinatorial procedure of merging two forecasts together.
           The forecasts often originate from various sources and thus have
        different values where they intersect. This sub-class is used to
        ensures the final forecasts does not show any misalignments.
        """
        indices:np.ndarray
        x_loc:np.ndarray
        y_loc:np.ndarray
        z_init:np.ndarray
        n:int
        m:int
        x_poly:Optional[np.ndarray]=None
        y_poly:Optional[np.ndarray]=None
        dist:Optional[np.ndarray]=None
        z_fc1:Optional[np.ndarray]=None
        z_fc2:Optional[np.ndarray]=None
        z_out:Optional[np.ndarray]=None
        def set_poly_points(self, poly:PolyCoords)->None:
            n_len = self.x_loc.shape
            self.x_poly = np.zeros(shape=n_len)
            self.y_poly = np.zeros(shape=n_len)
            self.dist   = np.ones(shape=n_len)*np.infty
            for i in range(self.x_loc.shape[0]):
                coord = Point(self.x_loc[i],self.y_loc[i])
                for line in poly.line_iterator():
                    dist = line.minimum_distance(coord=coord)
                    if dist < self.dist[i]:
                        self.dist[i] = dist
                        point_on_poly = line.point_on_line(coord)
                        self.x_poly[i] = point_on_poly[0]
                        self.y_poly[i] = point_on_poly[1]
            return None
        def interpolate(self, fc1:Forecast, fc2:Forecast):
            grid = GridCoords(x=self.x_poly, y=self.y_poly, proj=NotDefined())
            self.z_fc1 = fc1.interpolate_nearest_neighbour(grid=grid).z
            self.z_fc2 = fc2.interpolate_nearest_neighbour(grid=grid).z
        def calc_z_adjusted(self, epsilon):
            shape = self.x_loc.shape
            self.z_out = np.zeros(shape=shape)
            #print(np.max(self.dist))
            #print(epsilon)
            #epsilon = np.max(self.dist)
            for i in range(shape[0]):
                eps = max([epsilon,self.dist[i]])
                z1 = ((self.dist[i])/eps)*self.z_init[i]
                z2 = (1-(self.dist[i])/eps)*(self.z_fc1[i]+self.z_fc2[i])/2
                self.z_out[i] = z1 + z2

    @staticmethod
    def combine_subsets(
        ss1:Subset, ss2:Subset, proj:Projection, date:datetime.datetime,
        element:str, source:Union[str,List[str]] = 'unspecified'
    ) -> Forecast:
        x=np.concatenate([ss1.x_loc, ss2.x_loc])
        y=np.concatenate([ss1.y_loc, ss2.y_loc])
        z=np.concatenate([ss1.z_out, ss2.z_out])
        fc = Forecast(
            x=x, y=y, proj=proj, z=z,
            date=date, element=element, source=source
        )
        return fc

    def get_subset(self, poly:PolyCoords, inside=False) -> Forecast:
        """
        Use the polygon to get a subset of the forecast.
        If inside='True' then a subset will be created from the points which
        are inside the polygon. If inside='False' then the points inside the
        polygon will be removed and the points outside will be used to create
        the new dataset.
        -----------------------------------------------------------------------

        """
        x = self.x.flatten()
        y = self.y.flatten()
        z = self.z.flatten()

        poly_arr = poly.matrix
        if len(poly_arr.shape) !=2:
            raise ValueError()
        if poly_arr.shape[1] != 2:
            raise ValueError()
        poly_mpl = Path(poly.matrix)

        xy = np.vstack([x,y]).T
        xy_inside = poly_mpl.contains_points(xy)

        x = x[xy_inside==inside]
        y = y[xy_inside==inside]
        z = z[xy_inside==inside]

        return Forecast(
            x=x,y=y,z=z, source=self.source,
            date=self.date, proj=self.proj, element=self.element)

    def get_subset2(
        self, poly_inner:Optional[PolyCoords]=None,
        poly_outer:Optional[PolyCoords]=None
    ) -> SubForecast:
        """
        Use the polygon to get a subset of the forecast.
        If inside='True' then a subset will be created from the points which
        are inside the polygon. If inside='False' then the points inside the
        polygon will be removed and the points outside will be used to create
        the new dataset.
        -----------------------------------------------------------------------

        """
        if poly_inner==None and poly_outer==None:
            raise ValueError('both outer and inner polygons are equal to None')
        n,m = self.x.shape
        x = self.x.flatten()
        y = self.y.flatten()
        z = self.z.flatten()
        xy = np.vstack([x,y]).T

        if poly_inner!=None:
            poly_arr = poly_inner.matrix
            if len(poly_arr.shape) !=2:
                raise ValueError()
            if poly_arr.shape[1] != 2:
                raise ValueError()
            poly_mpl = Path(poly_inner.matrix)
            indices_inner = poly_mpl.contains_points(xy)
        if poly_outer!=None:
            poly_arr = poly_outer.matrix
            if len(poly_arr.shape) !=2:
                raise ValueError()
            if poly_arr.shape[1] != 2:
                raise ValueError()
            poly_mpl = Path(poly_outer.matrix)
            indices_outer = poly_mpl.contains_points(xy)==False
        if poly_inner!=None and poly_outer!=None:
            indices = indices_inner*indices_outer
        elif poly_inner!=None and poly_outer==None:
            indices = indices_inner
        elif poly_inner==None and poly_outer!=None:
            indices = indices_outer

        x = x[indices]
        y = y[indices]
        z = z[indices]

        return self.Subset(
            indices=indices, x_loc=x, y_loc=y, z_init=z, n=n, m=m
        )




    def interpolate_nearest_neighbour(self, grid:GridCoords, k=2) -> Forecast:
        """
        Use Clough Tocher interpolation to calculate the z-values corresponding
        to the locations in the grid.
        """
        if grid.x.shape[0]*grid.x.shape[0] <= 1:
            raise Exception(f"\nUnable to interpolate. \n{grid.x.shape=}\n{grid.y.shape=}")
        if self.x.shape!=self.y.shape or self.y.shape!=self.z.shape:
            raise Exception(f"\n{self.x.shape=}\n{self.y.shape=}\n{self.z.shape=}")

        x = self.x[np.isnan(self.z)==False].reshape([-1,1])
        y = self.y[np.isnan(self.z)==False].reshape([-1,1])
        z = self.z[np.isnan(self.z)==False].reshape([-1,1])
        coords = np.concatenate([x, y], axis=1)

        model = NearestNDInterpolator(
            coords, z
        )
        z_calc = model(grid.x, grid.y).squeeze()

        return Forecast(
            x=grid.x, y=grid.y, z=z_calc, date=self.date, proj=self.proj,
            element=self.element, source=self.source)

    def interpolate(self, grid:GridCoords, k=4) -> Forecast:
        def fix_directional_values(
            z:np.ndarray,
            units:Literal['radians','gradians'] = 'gradians'
        ) -> np.ndarray:
            interval = 0
            if units=='gradians':
                interval=360
            elif units=='radians':
                interval=np.pi
            else:
                raise ValueError(
                    "The input 'units' must be either be equal" +
                    "to 'radians' or 'gradians'" + f"\nNot {units}")
            z[interval <= z] -= interval
            z[z        <= 0] += interval
            return z

        if self.element == 'MWD':
            #return self.interpolate_nearest_neighbour(grid=grid, k=k)
            return self.interpolate_CloughTocher(grid=grid)
        elif self.element =='SWH':
            return self.interpolate_CloughTocher(grid=grid)
        raise ImplementationError(
            f"interpolation for forecasts of {self.element=} has "+
            "not been implemented"
        )

    def get_intersecting_polygons(self, polycords_co:PolyCoordsCollection) -> PolyCoordsCollection:
        """
        return the polygons in 'co_polys' which intersect with the Forecast
        """
        fc_Polygon = self.convexhull().GetShapelyPolygon()
        polys_co = polycords_co.GetShapelyPolygons()

        s = STRtree(polys_co)
        result = s.query(fc_Polygon)
        output = []
        for poly in result:
            #print(type(poly))
            poly_boundary = poly.boundary
            array = np.array(poly_boundary.coords)
            output.append(PolyCoords(array[:,0], array[:,1], proj=polycords_co.proj))
        return PolyCoordsCollection(polys=output, proj=polycords_co.proj)

    def get_intersect_coords(
        self, poly:PolyCoords, dist:int=10000
    ) -> List[Tuple[int]]:
        """
        Newest update.
        Function is not used. This feature was implemented through the subclass
        Forecast.Subset
        _______________________________________________________________________
        _______________________________________________________________________
        Original Note:
        Get the indices of the coordinates of the forecast which are a distance
        'dist' from the input polygon. Return the indices
        -----------------------------------------------------------------------
        Explanation of function use
        -----------------------------------------------------------------------
        Status Quo:
        For every two forecasts which are combined a polygon is created
        for the high resolution/quality forecast. This polygon is used to
        remove the points from the forecast of the lower quality. Afterwards,
        these forecasts are combined through interpolation onto a new grid.
        -----------------------------------------------------------------------
        Problem:
        Forecasts originating from different sources often have distinct values.
        This presents two difficulties.
        1. What Forecast is more reliable and thus, which should we more rely on
        2. How should we combine the forecasts. We don't want weather-forecasts
           with abrubt/intense weather-changes at the intersection of the two
           original forecasts.
        -----------------------------------------------------------------------
        This function will help solve these problem by identifying all the
        points within a distance γ from the polygon and locate the point on the
        polygon closest to the point from the forecast.
        This function will return the locations of each point located on the
        polygon and also the 1-dimensional indices of the points from the
        forecasts. If the forecasts have 2 dimensions for the x and y matrices.
        then these matrices will be flattened first.
        -----------------------------------------------------------------------
        Input:
            polygon:
            Forecast:
        Output:
            List[Tuple[index, dist_to_poly]]
        """
        print('WARNING: the class function Forecast.get_intersect_coords was used')
        x = self.x.flatten()
        y = self.y.flatten()
        output = []
        for i, point in enumerate(self.point_iterator()):
            min_dist = np.infty
            imin_dist = -1
            for j, line in enumerate(poly.line_iterator()):
                if np.abs(line.dist(point)) < min_dist and line.contains(point):
                    imin_dist = j
                    print(imin_dist)
        raise Exception()




    def fill_countries(self) -> Forecast:
        """
        Change the z-values which are located where countries are.
        If a grid of the same size and with the same locations has already been
        filled in, then this function will look it up and use it.
        Else, this function will perform this operation again and save the grid.
        """
        #----------------------------------------------------------------------
        # fill_countries: function 1
        def __calculate_grid_file_id_nr__(
                n:int, m:int, corner_ul:float, corner_lr:float) -> str:
            """ create a random index number, based on values from the input grid """
            return str(round(n*m+corner_ul+corner_lr))
        #----------------------------------------------------------------------
        # fill_countries: function 2
        def __fetch_cache__(
            self:Forecast,
            tempfile:str,
            country_value:int=100000
            ) -> Forecast:
            with open(tempfile, 'rb') as handle:
                points_in_country = pickle.load(handle)
            z = self.z.copy()
            z[points_in_country]=country_value
            return Forecast(
                x=self.x, y=self.y, z=z, proj=self.proj,
                date=self.date, element=self.element, source=self.source
            )
        #----------------------------------------------------------------------
        # fill_countries: function 3
        def __implementation__(
                self:Forecast,
                tempfile:str,
                country_value:int=100000
                ) -> Forecast:
            print(
                'A Forecast is being created with a new grid.',
                '\nThe countries need to be filled in'
            )

            polycoords:PolyCoordsCollection = fd.get_country_polygons()

            if polycoords.proj != self.proj:
                polycoords = polycoords.project(proj_target=self.proj)
            polys = self.get_intersecting_polygons(polycoords)

            n,m = self.z.shape
            points_in_country = np.zeros(shape=(n,m), dtype=bool)
            z = self.z.copy()
            for poly in polys:
                poly_arr=np.vstack([poly.x, poly.y]).T
                poly_path=Path(poly_arr)

                points = np.column_stack([
                    self.x.reshape([-1,1]),
                    self.y.reshape([-1,1])
                ])
                points_in_country:np.ndarray = poly_path.contains_points(points).reshape([n,m])
                z[points_in_country] = country_value

            pickle_data:np.ndarray = z==country_value
            try:
                with open(tempfile, 'wb') as handle:
                    pickle.dump(pickle_data, handle)
            except:
                raise Exception('Failed saving cache file')
            else:
                print(
                    'A cache file was saved into the ' +
                    f'local directory {gd.cache_dir}'
                )
            return Forecast(
                x = self.x, y=self.y, z=z, proj=self.proj,
                date=self.date, element=self.element, source=self.source)

        #----------------------------------------------------------------------
        # fill_countries: main
        n,m = self.x.shape
        file_ID = __calculate_grid_file_id_nr__(
            n=n, m=m, corner_ul=self.x[0,0], corner_lr=self.x[n-1,m-1])
        tempfile = f'{gd.cache_dir}/{file_ID}.p'
        if os.path.isfile(tempfile):
            return __fetch_cache__(self, tempfile=tempfile)
        else:
            return __implementation__(self, tempfile=tempfile)
    #--------------------------------------------------------------------------
    def increase_coastline_values(
            self, inc_value:int=10000, Meters_from_coast_line:int=3000
            ):
        """ Original Code from the first weather program """
        #----------------------------------------------------------------------
        # increase_coastline_values: function 1
        def __Get_spacing__(x_m, y_m, z):
            """ Original Code from the first weather program """
            x_min_meters = x_m.min()
            y_min_meters = y_m.min()
            x_max_meters = x_m.max()
            y_max_meters = y_m.max()
            x_spacing = abs(x_max_meters - x_min_meters)/z.shape[1]
            y_spacing = abs(y_max_meters - y_min_meters)/z.shape[0]
            spacing = (x_spacing + y_spacing)/2
            return spacing

        #----------------------------------------------------------------------
        # increase_coastline_values: function 2
        def __neighbors__(row, col, z):
            """ Original Code from the first weather program """
            for i in row-1, row, row+1:
                if i < 0 or i == len(z): continue
                for j in col-1, col, col+1:
                    if j < 0 or j == len(z[i]): continue
                    if i == row and j == col: continue
                    if ((i == row-1) or (i == row +1)) and ((j == col-1) or (j == col+1)): continue
                    yield (i,j)
        #----------------------------------------------------------------------
        # increase_coastline_values: main
        spacing = __Get_spacing__(self.x, self.y, self.z)

        Meters_of_each_cell = spacing
        number_of_cells = int(Meters_from_coast_line//Meters_of_each_cell)
        RememberChangedIndices = []
        z_out = self.z.copy()
        for i in range(number_of_cells):
            if i == 0:
                I, J = np.where(z_out >= inc_value)
                I = I.reshape([-1,1])
                J = J.reshape([-1,1])
                indices = list(map(tuple, np.concatenate([I,J], axis = 1)))
                RememberChangedIndices = []
            else:
                indices = copy.deepcopy(RememberChangedIndices)
                RememberChangedIndices = []
            for (row,col) in indices:
                for neighbor in __neighbors__(row, col, z_out):
                    r = int(neighbor[0])
                    c = int(neighbor[1])
                    if z_out[r,c] < inc_value:
                        z_out[r,c] = inc_value
                        if (r,c) not in RememberChangedIndices:
                            RememberChangedIndices.append((r,c))
        return Forecast(
            x=self.x, y=self.y, z=z_out, proj=self.proj, element=self.element,
            source=self.source, date=self.date)


    def transpose(self) -> Forecast:
        return Forecast(
            x=self.x.transpose(),
            y=self.y.transpose(),
            z=self.z.transpose(),
            proj=self.proj,
            date=self.date,
            element=self.element,
            source=self.source
        )

@dataclass(order=True)
class RasterArray:
    """
    Used to manage raster from Raster-Files (i.e. grib-files, nc-files, etc.)
    That is, individual forecasts for a specific date.
    z matrix with two dimensions of lengths NxM
    """
    z:np.ndarray
    date:datetime.datetime
    element:str="UnDefined"
    source:Union[str,List[str]] = 'UnDefined'
    def CreateForecastInstance(self, grid:GridCoords) -> Forecast:
        return Forecast(
            x=grid.x, y=grid.y, proj=grid.proj,
            z=self.z, date=self.date, element=self.element, source=self.source
        )




@dataclass(slots=True)
class ForecastCollection:
    """...
    This dataclass will be used to manage collection of multiple forecasts.
    - It will be used to manage the interpolation/combination of multiple
      forecasts to create a single forecast which we'll be able to use for
      route planning.
    """
    fcs:List[Forecast]
    sss:List[List[Tuple[int]]]
    #interp_grids:list[GridCoords]
    def __getitem__(self, index) -> Forecast:
        return self.fcs[index]
    def __iter__(self) -> Generator[Forecast,None,None]:
        yield from self.fcs

    @property
    def proj(self) -> Projection:
        return self.fcs[0].proj
    @property
    def element(self) -> str:
        return self.fcs[0].element
    @property
    def source(self) -> Union[List[str],str]:
        return self.fcs[0].source
    @property
    def date(self) -> datetime.datetime:
        return self.fcs[0].date


    def project(self, proj_target):
        pass

    def combine(self) -> Forecast:
        #sort(self.forecasts) by priority higher priority first
        n_fcs = len(self.fcs)
        if n_fcs <= 1:
            raise Exception("There must at least be two forecasts")
        Fc_out:Forecast = self.fcs[0]
        for i in range(1,n_fcs):
            polygon = Fc_out.convexhull()         # - get the polygon surrounding the yr_forecast
            polygon_inner = polygon.offset_polygon(
                dist=gd.intersection_distance, side='right')
            polygon_outer = polygon.offset_polygon(
                dist=gd.intersection_distance, side='left')

            fc = self.fcs[i]            # - get next forecast of lower priority
            fc_add = fc.get_subset(poly=polygon, inside=False)

            # fc_add_a: is the part of the forecast located on the intersection
            #   area. The points on this forecast need to be adjusted.
            # fc_add_b: is not within the intersection area, and thus does not
            #   need to be adjusted
            #fc_add_a = fc_add.get_subset(poly=polygon_outer, inside=True)
            #fc_add_b   = fc_add.get_subset(poly=polygon_outer, inside=False)
            fc_add_a:Forecast.Subset = fc.get_subset2(poly_inner=polygon_outer, poly_outer=polygon)
            fc_add_a.set_poly_points(poly=polygon.simplify())
            fc_add_a.interpolate(fc1=fc, fc2=Fc_out)
            fc_add_a.calc_z_adjusted(epsilon=gd.intersection_distance)

            fc_add_b:Forecast.Subset = fc.get_subset2(poly_inner=None, poly_outer=polygon_outer)
            fc_add_b.z_out = fc_add_b.z_init

            fc_add = Forecast.combine_subsets(
                ss1=fc_add_a, ss2=fc_add_b,
                proj=self.proj, date=self.date, element=self.element,
                source=self.source
            )

            Fc_a:Forecast.Subset = Fc_out.get_subset2(poly_outer=polygon_inner)
            Fc_a.set_poly_points(poly=polygon.simplify())
            Fc_a.interpolate(fc1=Fc_out, fc2=fc)
            Fc_a.calc_z_adjusted(epsilon=gd.intersection_distance)
            Fc_b = Fc_out.get_subset2(poly_inner=polygon_inner)
            Fc_b.z_out = Fc_b.z_init
            Fc_out = Forecast.combine_subsets(
                ss1=Fc_a, ss2=Fc_b,
                proj=self.proj, date=self.date, element=self.element,
                source=self.source
            )

            """
            For each point in fc_add_a and Fc_a, locate the point on the polygon
            corresponding to the point and calculate its z-value using both
            forecasts. When you have the z-values, then calculate the distance
            from the poly-point to the original point
            """

            #out1 = fc.get_intersect_coords(poly=polygon)
            #out2 = fcs_new.get_intersect_coords(poly=polygon)
            #self.sss.append(out1)
            #self.sss.append(out2)
            Fc_out = Fc_out + fc_add  # - combine the two forecasts
        #return Fc_out, polygon_inner, polygon_outer
        return Fc_out

    def increase_resolutions(self):
        """
        This function iterates over each forecast and increases their
        resolution through interpolation, such that all the forecasts have
        approximately the same resolution as the forecast with the highest
        resolution.
        (This is necessary due to the interpolation technique being used.)
        -----------------------------------------------------------------------
        This method should also be better because the area covered by each
        forecast can be interpolated more accurately. This is due to the fact
        that generally each forecast which is sourced is constructed on a grid.
        That is, the location of all the datapoints are stored in an NxM matrix
        and where their locations can be described with two one dimensional
        arrays (i.e. x & y) of length N and M. This allows for a more accurate
        interpolation for each area than it would if all the areas where first
        combined and then interpolated as part of a single set of datapoints.
        -----------------------------------------------------------------------
        Usually, the main problem which occurs when forecasts are combined is
        that the area in-between the forecasts looks very wierd, unatural.
        -----------------------------------------------------------------------
        To increase the resolution of each forecast we'll want to create a new
        grid based off the original grid of the forecast in its original
        projection.
        For each forecast we'll need to estimate its resolution (i.e. measure
        based of the distance between datapoints in the forecast). We'll
        compare this resolution to the forecast with the highest resolution
        with a multiplication factor. We'll use this multiplication factor to
        create a new grid in the forecasts initial projection.
        (NOTE: This will need to be updated later, such that unnecessary rep-
        etitive steps are eliminated.)
        """
        # as of now this function is implemented in "main_functions.py"
        pass





