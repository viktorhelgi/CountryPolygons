from main import main

if __name__ == '__main__':
    main(
        files = [
            'data/geoBoundaries/denmark/geoBoundaries-DNK-ADM1.shp',
            #'data/geoBoundaries/norway/geoBoundaries-NOR-ADM0.shp'
        ],
        plot_polygons=True,
        save_plot=True,
        save_data=False,
    )
