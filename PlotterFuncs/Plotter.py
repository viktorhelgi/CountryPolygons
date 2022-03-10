
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes._axes import Axes
from matplotlib import cm

from os import walk
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import pickle

# Local Modules
from .Colormap import Get_ColorMap
from .Get_Contour_Levels import Get_Contour_Levels
from .Project_Data import Project_Points
from .Polygons_Arrays_transformation import poly_arr_tr
from .Plot_Lat_Lon_Labels import GetLatLon_Labels, fmt

from .typehints import Country_Polygon
from typing import List, Union, Any, Optional, Literal, TypedDict
from shapely.geometry import MultiPolygon, Polygon

from mod.objects.Coordinates import PolyCoords
from mod.objects.Projections import (
    Projection, WorldGeodeticSystem, ObliqueMercator, UnknownProjection
)
from mod.objects.Forecasts import Forecast

class FunctionHasntBeenCalledError(Exception):
    """ E.g. before using Plot_method3.contourf(), first you need to call the class function Plot_method3.get_cmap() """
    pass
class CountryDict(TypedDict):
    boundaries:MultiPolygon
    main_land:Polygon
    polygons:List[np.ndarray]

from .data_info import output_dir, polygons_dir, polygons_dir_old, available_countries

from shapely.ops import unary_union

class Plotter:
    def __init__(
        self,
        rows = 1,
        cols = 1,
        forecast:Optional[Forecast]=None,
        colorbar = False,
        axes_on = True,
        maintain_xy_proportions = True,
        save_path = f"{output_dir}/Images_vol2/",
        image_name = None,
        figsize = 'NotDefined',
        maximize_plot=False,
        standardize_plots=True,
        cmap_z_min=0,
        cmap_z_max=10,
        input_based_xy_limits=True,
        new_method=True
        ):

        self.rows = rows
        self.cols = cols
        self.forecast = forecast
        self.colorbar = colorbar
        self.axes_on  = axes_on
        self.maintain_xy_proportions = maintain_xy_proportions
        self.save_path = save_path
        self.image_name = image_name
        self.my_cmap = 'coolwarm'
        self.maximize_plot = maximize_plot
        self.standardize_plots = standardize_plots
        self.input_based_xy_limits = input_based_xy_limits
        self.x_min = np.infty
        self.y_min = np.infty
        self.x_max = -np.infty
        self.y_max = -np.infty
        self.new_method = new_method

        self.z_bounds = {
            'SWH':{'min':0, 'max':10},
            'MWD':{'min':0, 'max':360}
        }
        if self.forecast==None:
            if self.standardize_plots:
                self.set_cmap(z_min=cmap_z_min, z_max=cmap_z_max)
        else: # that is if Forecast != None
            self.input_based_xy_limits=False
            self.maximize_plot=True
            self.standardize_plots=True
            self.element = self.forecast.element
            if type(self.element) != str:
                raise TypeError(f"{type(self.element) = }")
            if self.element not in self.z_bounds.keys():
                raise NotImplementedError(
                    "The Plotter function hasn't been configured for a " +
                    "forecast with an element of '{element}'"
                )
            z_min = self.z_bounds[self.element]['min']
            z_max = self.z_bounds[self.element]['max']
            self.set_cmap(z_min=z_min, z_max=z_max)


        if figsize == 'NotDefined':
            subplots_obj = plt.subplots(rows, cols)
            self.fig:plt.Figure = subplots_obj[0]
            self.ax = subplots_obj[1]
        else:
            assert type(figsize) == tuple, 'figsize needs to be a tuple'
            assert len(figsize) == 2, 'figsize needs to be a tuple of lenght 2'
            #self.fig, self.ax = plt.subplots(rows, cols)
            self.fig:plt.Figure = plt.figure(figsize = figsize)
            self.ax = self.fig.subplots(nrows=rows, ncols=cols)





    #--------------------------------------------------------------------------
    def set_title(self, title:str) -> None:
        self.fig.canvas.set_window_title(title)
        return None

    def set_subtitle(self, title:str, row:int, col:int) -> None:
        ax = self.get_ax(row, col)
        ax.title.set_text(title)
        return None

    def get_ax(self, row:int=1, col:int=1) -> plt.Axes:
        #if self.rows == 1 and self.cols == 1:
        #    return self.ax
        #elif (self.rows == 1 and self.cols != 1) or (self.rows != 1 and self.cols == 1):
        if row <= 0 or col <= 0:
            raise ValueError(f"either row or col are negative or 0 \n{row=}\n{col=}")
        if (self.rows == 1 and self.cols != 1) or (self.rows != 1 and self.cols == 1):
            return self.ax[max(row-1, col-1)]
        elif (self.rows != 1 and self.cols != 1):
            return self.ax[row-1, col-1]
        return self.ax

    def set_cmap(
            self, z=None, level_res=1,
            z_min:float=0, z_max:float=5, n_levels=80,
            colormap_name:Literal[
                'magma', 'inferno', 'plasma', 'viridis', 'cividis',
                'twilight', 'twilight_shifted', 'turbo'
            ] = 'turbo'
        ) -> None:
        if self.standardize_plots:
            # levels on the colormap
            # Available Colormaps are
            # >> cmaps = ['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'twilight_shifted', 'turbo']
            n_levels+=1
            colormap = cm.get_cmap(colormap_name, n_levels)
            my_colors:np.ndarray = colormap.colors

            # This ensures, that the color of the final level of the colormap
            # will be white. This is the color of the countries
            #under_0 = np.zeros(my_colors.shape[1]).reshape([1,-1])
            on_land = np.ones(my_colors.shape[1]).reshape([1,-1])
            my_colors = np.concatenate([my_colors, on_land])
            self.my_cmap = mpl.colors.ListedColormap(my_colors)

            # my norm
            colors_intervals = np.linspace(z_min, z_max, n_levels).tolist()

            #colors_intervals.append(100000)
            self.my_norm = mpl.colors.BoundaryNorm(
                colors_intervals,
                n_levels)
        else:
            if z==None:
                raise ValueError(f"The value of z is [={z}]")
            self.my_cmap, self.my_norm = Get_ColorMap(
                z,
                increase_resolution_const = level_res
            )


    def scatter(
        self, x:np.ndarray, y:np.ndarray, z:np.ndarray=np.array([]),
        row:int=1, col:int=1, pt_size:float=10.0, enumerate_pts:bool=False,
        fontsize:int=2, color:Literal['b','g','r','c','m','y','k','w']='k',
        enumerate_method1:bool = False
    ):
        ax = self.get_ax(row, col)
        if self.new_method:
            pass
        elif self.input_based_xy_limits:
            self.x_min = np.min([self.x_min, np.min(x)])
            self.y_min = np.min([self.y_min, np.min(y)])
            self.x_max = np.max([self.x_max, np.max(x)])
            self.y_max = np.max([self.y_max, np.max(y)])

        if self.new_method:
            if 0 == len(z):
                scatter_plot = ax.scatter(
                    x, y,
                    s=pt_size,
                    c=color
                )
            else:
                scatter_plot = ax.scatter(
                    x, y,
                    s=pt_size,
                    c=z,
                    cmap=self.my_cmap, norm=self.my_norm
                )
        elif self.standardize_plots:
            scatter_plot = ax.scatter(
                x, y,
                s=pt_size,
                c=z,
                cmap=self.my_cmap, norm=self.my_norm
            )
        elif enumerate_method1:
            scatter_plot = ax.scatter(x,y,s = pt_size, c = color)
        else:
            scatter_plot = ax.scatter(x,y,s = pt_size, c = z)
        if self.colorbar:
            self.fig.colorbar(scatter_plot)


        if self.axes_on == False:
            ax.set_axis_off()
        if self.maintain_xy_proportions:
            ax.set_aspect(1/ax.get_data_ratio())

        if enumerate_method1 == True and enumerate_pts:
            for i in range(z.shape[0]):
                value = z[i]
                x_pt = x[i]+0.05
                y_pt = y[i]
                ax.annotate(value, (x_pt,y_pt), color = color, size = fontsize)
        elif enumerate_pts:
            if len(x.shape) == 1:
                for i, txt in enumerate(list(z)):
                    print(txt)
                    ax.annotate(round(y[i],3), (x[i], y[i]))
            else:
                for i in range(z.shape[0]):
                    for j in range(z.shape[1]):
                        txt = z[i,j]
                        ax.annotate("x : {}".format(str(int(x[i,j]))), (x[i,j], y[i,j] + 12000), fontsize = fontsize, color = color)
                        ax.annotate("y : {}".format(str(int(y[i,j]))), (x[i,j], y[i,j] + 7000), fontsize = fontsize, color = color)
                        ax.annotate("z : {}".format(round(txt,3)), (x[i,j], y[i,j]+2000), fontsize = fontsize, color = color)

    def line(
        self, x, y, row = 1, col = 1, linewidth = 2,
        color:Literal['b','g','r','c','m','y','k','w']='m',
        linestyle:Literal['-','--','-.',':']='-',
        marker:Literal['','v','>','<']=''
    ) -> None:
        ax = self.get_ax(row, col)
        ax.plot(x,y, linewidth = linewidth, color=color, linestyle=linestyle)

        if self.axes_on == False:
            ax.set_axis_off()

    def matrix(self, z, row = 1, col=1, upper_limit=100000):
        plt.imshow(z, cmap='coolwarm', vmax=upper_limit)

    def plot_iceland(self, proj_target, row = 1, col = 1):
        ax = self.get_ax(row, col)
        pickle_file = '/home/vilhjalmur/vedur/Data/2_Helper_files/Iceland/Polygons/ISL_ADM2_fixedInternalTopology_polygons.p'
        with open(pickle_file, 'rb') as handle:
            dat = pickle.load(handle)
            xy = poly_arr_tr(dat['Main_Land'])
            proj_source = dat['proj_wkt']

            data_projected = Project_Points(xy, proj_source = proj_source, proj_target = proj_target)
            xy_projected = data_projected['array']

            x_ice = xy_projected[:,0]
            y_ice = xy_projected[:,1]
        ax.plot(x_ice, y_ice)

    def plot_country(
        self,
        proj_target:Projection=UnknownProjection(type='EPSG', ProjectionRef=3857),
        country:available_countries='undefined',
        country_data:Optional[List[np.ndarray]]=None,
        row = 1,
        col = 1,
        color = 'grey',
        linewidth = 0.5
    ) -> None:
        """ vol4 """
        ax = self.get_ax(row, col)
        if type(country_data).__name__=='NoneType':
            if country not in ['denmark', 'norway']:
                raise ValueError('The input country must be either denmark or norway')
            pickle_file = f'{polygons_dir}/{country}.p'      # country: {Denmark, Great_Britain, Greenland, Ic
            with open(pickle_file, 'rb') as handle:
                data:CountryDict = pickle.load(handle)
                polygons:List[np.ndarray] = data['polygons']
        else:
            if type(country_data) != list:
                raise TypeError(
                    f'The input should be a list, not {type(country_data)=}')
            if type(country_data[0]) != np.ndarray:
                raise TypeError(
                    f'The input should be a np.ndarray, not {type(country_data[0])=}')
            polygons:List[np.ndarray]=country_data

        #Multipolygon: MultiPolygon = MultiPolygon()
        #if type(list(polygons)[0]).__module__ == 'numpy':
        #    Multipolygon = poly_arr_tr(obj_in = polygons, output_type = 'multipolygon')
        #print('  CP: Nr of Polygons before combination: ', len(Multipolygon.geoms))
        #Multipolygon_only_Borders = unary_union(Multipolygon)
        #print('  CP: Nr of Polygons after combination: ', len(Multipolygon_only_Borders.geoms))
        #polygons = poly_arr_tr(obj_in=Multipolygon_only_Borders, output_type='numpy')


        for poly in polygons:
            poly_p = PolyCoords(
                x=poly[:,0],
                y=poly[:,1],
                proj=WorldGeodeticSystem()
            ).project(proj_target=proj_target)
            ax.plot(poly_p.x, poly_p.y, color=color, linewidth=linewidth)
        return None
    def plot_country3(self, proj_target, country, row = 1, col = 1, color = 'grey', linewidth = 0.5) -> None:
        """ vol3 """
        ax = self.get_ax(row, col)
        pickle_file = f'C:/Users/Lenovo/Documents/Coding/Hefring/NoDkferd/data/pickles/{country}_PolyCoords.p'      # country: {Denmark, Great_Britain, Greenland, Ic
        with open(pickle_file, 'rb') as handle:
            data = pickle.load(handle)
        for poly in data['polys']:
            poly_p = PolyCoords(
                x=poly['x'],
                y=poly['y'],
                proj=WorldGeodeticSystem()
            ).project(proj_target=proj_target)
            ax.plot(poly_p.x, poly_p.y, color=color, linewidth=linewidth)
        return None

    def plot_country_vol2(self, proj_target, country, row = 1, col = 1, color = 'grey', linewidth = 0.5) -> None:

        ax = self.get_ax(row, col)
        pickle_file = f'{polygons_dir_old}/{country}_polygons.p'      # country: {Denmark, Great_Britain, Greenland, Ic
        with open(pickle_file, 'rb') as handle:
            data:Country_Polygon = pickle.load(handle)
            polys:List[np.ndarray] = data['boundaries']['polygons']
            proj_source:Union[str,int]  = data['boundaries']['proj_wkt']
        for poly in polys:
            poly_p_dict = Project_Points(
                poly, proj_source=proj_source, proj_target=proj_target)
            poly_p = poly_p_dict['array']
            x_coords = poly_p[:,0]
            y_coords = poly_p[:,1]
            ax.plot(x_coords, y_coords, color=color, linewidth=linewidth)
        return None


    def plot_country_vol1(self, proj_target, country, row = 1, col = 1, color = 'grey', linewidth = 0.5):
        ax = self.get_ax(row, col)
        pickle_file = 'C:/Users/Lenovo/Documents/Coding/Hefring/January/Data/pickle_files/{}_polygons.p'.format(country)      # country: {Denmark, Great_Britain, Greenland, Ic

        with open(pickle_file, 'rb') as handle:
            dat:Country_Polygon = pickle.load(handle)

            main_land = dat['Main_Land']

            country_polygon = poly_arr_tr(main_land)
            proj_source = dat['proj_wkt']
            country_dict = Project_Points(country_polygon, proj_source = proj_source, proj_target = proj_target)
            country_array = country_dict['array']
            x_coords = country_array[:,0]
            y_coords = country_array[:,1]
        ax.plot(x_coords, y_coords, color = color, linewidth = linewidth)

    def contourf(self, x, y, z, row = 1, col = 1, Standard_Contourf_Levels = True, nr_of_levels = 30, level_res = 1, alpha = 1, seastate = False):
        ax = self.get_ax(row, col)
        if self.axes_on == False:
            ax.set_axis_off()
        if self.input_based_xy_limits:
            self.x_min = np.min([self.x_min, np.min(x)])
            self.y_min = np.min([self.y_min, np.min(y)])
            self.x_max = np.max([self.x_max, np.max(x)])
            self.y_max = np.max([self.y_max, np.max(y)])

        if self.standardize_plots:
            ax.contourf(x,y,z, levels=self.my_norm.boundaries, norm = self.my_norm, cmap=self.my_cmap, extend = 'max', alpha = alpha)
        elif seastate:
            Levels = Get_Contour_Levels(z, levels = nr_of_levels)
            norm = mpl.colors.Normalize(vmin=min(Levels), vmax=max(Levels))
            ax.contourf(x,y,z, levels = Levels, extend = 'max', alpha = alpha,norm = norm, cmap='Oranges')
        elif Standard_Contourf_Levels:
            if hasattr(self, 'my_norm') is False:
                raise FunctionHasntBeenCalledError('self.my_norm has not been defined call the function self.set_cmap()')
            ax.contourf(x,y,z, levels = self.my_norm.boundaries[:len(self.my_norm.boundaries) -3], norm = self.my_norm, cmap=self.my_cmap, extend = 'max', alpha = alpha)

        else:
            Levels = Get_Contour_Levels(z)
            norm = mpl.colors.Normalize(vmin=min(Levels), vmax=max(Levels))
            ax.contourf(x,y,z, levels = Levels, extend = 'max', alpha = alpha, norm = norm, cmap='coolwarm')

    def scatter_errors(self, row:int=1, col:int=1):
        if type(self.forecast)!=Forecast:
            raise TypeError("The input isn't a Forecast")

        z_min = self.z_bounds[self.element]['min']
        z_max = self.z_bounds[self.element]['max']
        ax = self.get_ax(row, col)
        x = self.forecast.x.flatten()
        y = self.forecast.y.flatten()
        z = self.forecast.z.flatten()

        out_of_bounds = ((z_min<=z)*(z<=z_max)==False) * (z!=100000)
        x = x[out_of_bounds]
        y = y[out_of_bounds]
        z = z[out_of_bounds]
        ax.scatter( x, y, s=13, color='red')



    def plot_latlon_lines(self, x_m, y_m, x_l, y_l, proj, row = 1, col = 1, font_size = 3, linewidths = 0.4, colors = 'darkblue', Location = 'undefined'):
        ax = self.get_ax(row, col)
        L_x, L_y, Loc_x, Loc_y = GetLatLon_Labels(
            x_l = x_l,
            y_l = y_l,
            Location = Location,
            false_easting = 0,
            false_northing = 0,
            proj_target = proj
        )

        LonLines = plt.contour(x_m, y_m, x_l, levels = L_x, linewidths = linewidths, colors = colors, linestyles = 'dashed',)
        Label1 = ax.clabel(LonLines,  fontsize= font_size, fmt = fmt, use_clabeltext = True, manual = Loc_x)

        LatLines = plt.contour(x_m, y_m, y_l, levels = L_y, linewidths = linewidths, colors = colors, linestyles = 'dashed')
        Label2 = ax.clabel(LatLines,  fontsize = font_size, fmt = fmt, use_clabeltext = True, manual = Loc_y)

        for l in Label1+Label2:
            l.set_rotation(0)

        if self.axes_on == False:
            ax.set_axis_off()

    def add_colorbar(self, row:int=1, col:int=1, ticks:Optional[np.ndarray]=None) -> None:
        ax = self.get_ax(row, col)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = mpl.colorbar.Colorbar(cax, cmap=self.my_cmap, norm=self.my_norm, orientation='vertical', ticks=ticks)
        return None


    """
    Deprecated
    """
    def create_Colorbar(self, row=1, col=1):
        raise NotImplementedError("This Function doesn't work")
        ax = self.get_ax(row, col)
        #plt.colorbar(cm.ScalarMappable(norm=self.my_norm, cmap=self.my_cmap), orientation='vertical')
        #ax.colorbar()
        ax.axes.box_aspect = 10


    def set_limits(self, x_min, y_min, x_max, y_max, row = 1, col = 1):
        """ set limits simply given the boundaries x-y min and max """
        ax = self.get_ax(row, col)
        self.input_based_xy_limits=False
        try:
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
        except:
            raise ValueError(
                f'Error performing >> ax.set_xlim([...]), for {x_min=}  {x_max=}',
                f'  {y_min=}  {y_max=}'
            )

    def set_limits_method2(self, x, y, radius, row = 1, col = 1):
        """ set the limits centered around the point (x,y). Ensures the x and y axis are the same length. """
        ax = self.get_ax(row, col)
        ax.set_xlim([x-radius, x+radius])
        ax.set_ylim([y-radius, y+radius])


    def savefig(self, dpi = 1000, quality = 100, optimize = True, image_name='.png'):
        self.get_image_path()
        if self.input_based_xy_limits:
            for row in range(self.rows):
                for col in range(self.cols):
                    self.set_limits(
                        col=col+1, row=row+1,
                        x_min=self.x_min, y_min=self.y_min,
                        x_max=self.x_max, y_max=self.y_max
                    )
        if self.maintain_xy_proportions:
            for ax in self.axes:
                ax.set_aspect(1/ax.get_data_ratio())

        if image_name != '.png':
            plt.savefig(image_name, bbox_inches='tight', pad_inches=0, dpi = dpi)#, quality = quality, optimize = optimize)#,pil_kwargs={'optimize':optimize, 'dpi':dpi, 'quality':quality})
        elif '.png' not in self.file_path:
            print('- Path to Image: ', self.file_path + '.png')
            plt.savefig(self.file_path, bbox_inches='tight', pad_inches=0, dpi = dpi)#, quality = quality, optimize = optimize)#,pil_kwargs={'optimize':optimize, 'dpi':dpi, 'quality':quality})
        else:
            print('- Path to Image: ', self.file_path)
            plt.savefig(self.file_path, bbox_inches='tight', pad_inches=0, dpi = dpi)#, quality = quality, optimize = optimize)#,pil_kwargs={'optimize':optimize, 'dpi':dpi, 'quality':quality})

    def get_image_path(self):
        if self.image_name == None:
            created_images = []
            for _, _, created_images in walk(self.save_path):
                pass
            max_volume = 0
            for image in created_images:
                if 'vol_' in image:
                    vol_nr = image.split('.png')[0].split('vol_')[-1]
                    max_volume = max(max_volume, int(vol_nr))
            self.image_name = 'Image_vol_{}.png'.format(str(max_volume+1))

        if self.save_path[-1] != '/':
            self.save_path = self.save_path + '/'
        self.file_path = self.save_path + self.image_name

    def show(self):
        if self.maximize_plot:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        if self.input_based_xy_limits:
            for row in range(self.rows):
                for col in range(self.cols):
                    self.set_limits(
                        col=col+1, row=row+1,
                        x_min=self.x_min, y_min=self.y_min,
                        x_max=self.x_max, y_max=self.y_max
                    )
        if self.maintain_xy_proportions:
            for ax in self.axes:
                ax.set_aspect(1/ax.get_data_ratio())
        plt.show()

    @property
    def axes(self):
        return [
            self.get_ax(row+1,col+1) for row in range(self.rows) for col in range(self.cols)
        ]



    def INITIALIZE(self, proj_target, row:int=1, col:int=1):
        """ functions specifically created for this problem. """
        ax = self.get_ax(row, col)
        self.plot_country(proj_target=proj_target, country='denmark')
        self.plot_country(proj_target=proj_target, country='norway')
        self.set_limits(
            x_min=9.7998e6, x_max=9.9311e6,
            y_min=1.2126e7, y_max=1.2510e7
        )





