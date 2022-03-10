
# updated: 17.2.2022
#------------------------------------------------------------------------------
from __future__ import annotations # used for advanced typehinting
#------------------------------------------------------------------------------
from dataclasses import dataclass, field
import numpy as np
from osgeo import osr
#------------------------------------------------------------------------------
from mod.objects.Coordinates import LineCoords
from mod.objects.Projections import Projection, ProjectionRefDefinition
#------------------------------------------------------------------------------

@dataclass
class RouteCoords(LineCoords):
    def project(self, proj_target:Projection) -> RouteCoords:
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
            x_pt = self.x[pt_i]
            y_pt = self.y[pt_i]
            pts_projected = trfm.TransformPoint(
                x_pt,y_pt)[:2]
            x_tg[pt_i] = pts_projected[0]
            y_tg[pt_i] = pts_projected[1]
        return RouteCoords(x=x_tg, y=y_tg, proj=proj_target)


