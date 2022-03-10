from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Optional, Generator
from matplotlib.path import Path
from scipy.interpolate import CloughTocher2DInterpolator
from shapely.strtree import STRtree

import numpy as np
import datetime


from CoordinateObjects import GridCoords, PolyCoords, PolyCoordsCollection,

class __ForecastInputError__(Exception):
    """
    The three matrices, x,y, and z
    """
    pass
class __ForecastCombinationError__(Exception):
    pass

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
    element:Optional[Union[List[str], str]]
    # Default Valued Variables
    source:Union[str,List[str]] = 'unspecified'
    """ ... add later
    spacing:float = field(init=False, metadata={'units':'degrees'})
    priority:???
    """

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

    def interpolate(self, grid:GridCoords) -> Forecast:
        """
        Use Clough Tocher interpolation to calculate the z-values corresponding
        to the locations in the grid.
        """
        if grid.x.shape[0]*grid.x.shape[0] <= 1:
            raise Exception("\nUnable to interpolate. \n{grid.x.shape=}\n{grid.y.shape=}")
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
        z_calc = model(grid.x, grid.y)
        return Forecast(
            x=grid.x, y=grid.y, z=z_calc,
            date=self.date, proj=self.proj, element=self.element)
    def fill_countries(self, co_polys:PolyCoordsCollection) -> Forecast:
        """
        Change the z-values which are located where countries are.
        """
        print(co_polys[0])
        fc_Polygon = self.convexhull().GetShapelyPolygon()
        co_Polygons = co_polys.GetShapelyPolygons()

        print(type(co_Polygons))
        s = STRtree(co_Polygons)
        result = s.query(fc_Polygon)
        print(result)
        print(type(result))

        for i, poly in enumerate(co_Polygons):
            if result[i]:
                continue

        return Forecast(
            x=self.x,
            y=self.y,
            z=self.z,
            proj=self.proj,
            date=self.date,
            element=self.element,
            source=self.source
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
    #interp_grids:list[GridCoords]
    def __getitem__(self, index) -> Forecast:
        return self.fcs[index]
    def __iter__(self) -> Generator[Forecast,None,None]:
        yield from self.fcs
    def project(self, proj_target):
        pass
    def combine(self) -> Forecast:
        #sort(self.forecasts) by priority higher priority first
        n_fcs = len(self.fcs)
        if n_fcs <= 1:
            raise Exception("There must at least be two forecasts")
        fcs_new:Forecast = self.fcs[0]
        for i in range(1,n_fcs):
            #if self.fcs[
            fc = self.fcs[i]            # - get next forecast of lower priority
            polygon = fcs_new.convexhull()         # - get the polygon surrounding the yr_forecast
            fc = fc.get_subset(      # - create a subset from the forecast from the points
                poly=polygon, inside=False)            #   on the outside of the polygon
            fcs_new = fcs_new + fc  # - combine the two forecasts
        return fcs_new

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






