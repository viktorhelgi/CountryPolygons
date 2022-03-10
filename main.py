import pickle
from typing import Union, List, Dict
from Create_Country_Polygons.create_country_polygons import create_country_polygons
from PlotterFuncs.Plotter import Plotter


def main(
    files:Union[List[str],str] = 'undefined',
    plot_polygons:bool = False,
    save_plot:bool = False,
    proj_target:str = '+proj=utm +zone=27',
    lenght_of_interval:int = 10000,
    save_data=False
):
    if files == 'undefined':
        files = ['C:/Users/Lenovo/Documents/Coding/Hefring/January/data/geoBoundaries-NOR-ADM0-all/geoBoundaries-NOR-ADM0.shp']
    nr_of_countries = len(files)
    print('Number of shp-files: {}'.format(nr_of_countries))
    for index, file in enumerate(files):
        print('/n################################################')
        print('# Iteration: {} of {}'.format(index+1, nr_of_countries))
        print('################################################')

        output = create_country_polygons(
            GeoBoundary_file=file,
            plot_polygons=plot_polygons,
            proj_target = proj_target,
            lenght_of_interval = 10000
        )
        info = file.split('/')
        country_name = info[len(info)-2].split('.')[0]
        print('create_country_polygons done')
        if save_data:
            file = 'data/pickles/{}.p'.format(country_name)
            with open(file, 'wb') as handle:
                pickle.dump(output, handle)
        if plot_polygons:
            obj_plot = Plotter(
                rows=1, cols=2,
                axes_on=True, input_based_xy_limits=False,
            )

            obj_plot.set_subtitle(row=1, col=1, title='country_borders')
            obj_plot.set_subtitle(row=1, col=2, title='polygons')

            obj_plot.plot_country(
                row=1, col=1,
                country_data=output['country_borders'])
            obj_plot.plot_country(
                row=1, col=2,
                country_data=output['polygons'])
            if save_plot:
                obj_plot.savefig(image_name=f'{country_name}.png')
            else:
                obj_plot.show()





if __name__ == '__main__':
    main(
        files = [
            'data/geoBoundaries/denmark/geoBoundaries-DNK-ADM1.shp',
            'data/geoBoundaries/norway/geoBoundaries-NOR-ADM0.shp'
        ],
        plot_polygons=False,
        save_data=True,
        proj_target = '+proj=utm +zone=27',
        lenght_of_interval = 10000
    )




