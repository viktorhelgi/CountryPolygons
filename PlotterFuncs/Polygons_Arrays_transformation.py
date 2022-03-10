"""
Module: Polygons_Arrays_transformation.py
----------------------------------
Version 2
Module has been updated to work with Shapely 2.0.
https://shapely.readthedocs.io/en/latest/migration.html#creating-numpy-arrays-of-geometry-objects
-----------------------------------
Only use the function 'poly_arr_tr'
>> from Polygons_Arrays_transformation import poly_arr_tr
-----------------------------------
Information about 'poly_arr_tr':
 Input can be one of the following:
 - np.ndarray                          - single geometric object
 - shapely.geometry.polygon            - single geometric object
 - shapely.geometry.linestring         - single geometric object
 or
 - list of np.ndarray's                            - multiple geometric objects
 - list of shapely.geometry.polygon's              - multiple geometric objects
 - list of shapely.geometry.linestring's           - multiple geometric objects
 - list of lists containing tuples                 - multiple geometric objects
 - list of lists ... of lists containing tuples    - multiple geometric objects
 - shapely.geometry.multipolygon                   - multiple geometric objects
 - shapely.geometry.multilinestring                - multiple geometric objects

 Output is specified as either geom_obj or list or numpy
 geom_obj will return one of the following:
 - shapely.geometry.linestring
 - shapely.geometry.polygon
 - shapely.geometry.multilinestring
 - shapely.geometry.multipolygon
 list will return the following:
 - list of np.ndarrays
-----------------------------------
run file, will test it
"""


import typing as tp

from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
import numpy as np

import sys

import matplotlib.pyplot as plt


def poly_arr_tr(obj_in:tp.Any, output_type = 'numpy', testing = False) -> tp.Any: #tp.Union[Polygon, MultiPolygon, LineString, MultiLineString, tp.List]:

    #####################################################
    # Procedure:
    # 1. determine input type and convert it to a list of np.ndarrays.
    # 2. Convert the list of np.ndarrays to the requested output.

    ############################
    #           Code           #
    ############################

    # Figure out the type of the input:

    # first if
    # - list
    # - shapely
    # - np.ndarray

    ###############
    # Procedure 1 #
    ###############

    return_list = True
    obj_numpy = np.array([])


    if type(obj_in).__module__ == 'numpy':
        print(" - type(obj_in).__module__ == 'numpy'") if testing else None
        obj_numpy = [obj_in]
        return_list = False

    elif type(obj_in) == list:                    # Level 1: list of ######
        print(" - type(obj_in) == list") if testing else None

        if type(obj_in[0]) == tuple:                                               # Level 2: list of tuples
            print("   - type(obj_in) == tuple") if testing else None
            obj_numpy = np.array(obj_in)

            if obj_numpy.shape[0] == 2:
                n,m = obj_numpy.shape
                obj_numpy = obj_numpy.reshape([m,n])
            assert len(obj_numpy.shape) == 2
            assert obj_numpy.shape[1] == 2
            obj_numpy = [obj_numpy]

        elif type(obj_in[0]) == list:                                              # Level 2: list of lists
            print("   - type(obj_in) == list") if testing else None
            obj_numpy = []
            for index, item in enumerate(obj_in):
                assert type(item[0]) == tuple
                arr = np.array(item)

                if arr.shape[0] == 2:
                    n,m = arr.shape
                    arr = arr.reshape([m,n])
                assert len(arr.shape) == 2
                assert arr.shape[1] == 2
                obj_numpy.append(arr)

        elif type(obj_in[0]).__module__ == 'numpy':                                # Level 2: list of np.ndarrays
            print("   - type(obj_in).__module__ == 'numpy'") if testing else None
            obj_numpy = obj_in

        elif type(obj_in[0]).__module__ == 'shapely.geometry.polygon':             # Level 2: list of polygons
            print("   - type(obj_in).__module__ == 'shapely.geometry.polygon'") if testing else None
            obj_numpy = [np.array(xy.boundary) for xy in obj_in]

        elif type(obj_in[0]).__module__ == 'shapely.geometry.linestring':          # Level 2: list of linestrings
            print("   - type(obj_in).__module__ == 'shapely.geometry.linestring'") if testing else None
            obj_numpy = [np.array(xy.coords) for xy in obj_in]

    elif type(obj_in).__module__.split('.')[0] == 'shapely':                   # Level 1: shapely object
        print(" - type(obj_in).__module__.split('.')[0] == 'shapely'") if testing else None

        if type(obj_in).__module__ == 'shapely.geometry.polygon':                  # Level 2: shapely.geometry.polygon
            print("   - type(obj_in).__module__ == 'shapely.geometry.polygon'") if testing else None
            obj_in2 = obj_in.boundary
            obj_in3 = np.array(obj_in2.coords)
            obj_numpy = [obj_in3]
            return_list = False

        elif type(obj_in).__module__ == 'shapely.geometry.linestring':             # Level 2: shapely.geometry.linestring
            print("   - type(obj_in).__module__ == 'shapely.geometry.linestring'") if testing else None
            obj_numpy = [np.array(obj_in.coords)]
            return_list = False


        elif type(obj_in).__module__ == 'shapely.geometry.multipolygon':           # Level 2: shapely.geometry.multipolygon
            print("   - type(obj_in).__module__ == 'shapely.geometry.multipolygon'") if testing else None
            obj_numpy =[]
            for obj_item in obj_in.geoms:
                obj_numpy.append(np.array(obj_item.boundary.coords))

        elif type(obj_in).__module__ == 'shapely.geometry.multilinestring':        # Level 2: shapely.geometry.multilinestring
            print("   - type(obj_in).__module__ == 'shapely.geometry.multilinestring'") if testing else None
            obj_numpy =[]
            for obj_item in obj_in.geoms:
                obj_numpy.append(np.array(obj_item.coords))

    type_obj_numpy_list_processed = type(obj_numpy)
    assert type_obj_numpy_list_processed == list
    type_obj_numpy_element_processed = type(obj_numpy[0])
    assert type_obj_numpy_element_processed.__module__ == 'numpy'




    if 'numpy' in output_type:
        print(" - 'numpy' in output_type") if testing else None
        obj_out = obj_numpy
    elif 'polygon' in output_type or 'multipolygon' in output_type:
        print("   - 'polygon' in output_type or 'multipolygon' in output_type") if testing else None
        for poly in obj_numpy:
            first_pt = poly[0,:]
            last_pt  = poly[-1,:]
            assert (first_pt == last_pt).all()

        obj_out = [Polygon(list(map(tuple,arr))) for arr in obj_numpy]

        if 'multipolygon' in output_type:
            print(" - 'multipolygon' in output_type") if testing else None
            obj_out = MultiPolygon(obj_out)

    elif 'linestring' in output_type or 'multilinestring' in output_type:
        print(" - output_type == 'linestring' or output_type == 'multilinestring'") if testing else None
        obj_out = [LineString(list(map(tuple,arr))) for arr in obj_numpy]
        if 'multilinestring' in output_type:
            print("   - 'multilinestring' in output_type") if testing else None
            obj_out = MultiLineString(obj_out)



    if return_list == False:
        assert type(obj_out) == list
        assert len(obj_out) == 1
        obj_out = obj_out[0]
    return obj_out

def test_Polygon_to_Array(testing = True):
    poly_in = Polygon(((0,4),(1,3),(19,1),(2,3),(9,1),(4,1),(0,4)))

    print('\nInput: {}\n{} \n'.format(type(poly_in), poly_in))

    numpy_out = poly_arr_tr(
        obj_in = poly_in,
        output_type= 'numpy',
        testing = testing
    )

    print('\nOutput: {}\n{} \n'.format(type(numpy_out), numpy_out))


def test_Linestring_to_Array(testing = True):
    line_in = LineString(((0,4),(1,3),(19,1),(2,3),(9,1),(4,1),(0,4)))

    print('\nInput: {}\n{} \n'.format(type(line_in), line_in))

    numpy_out = poly_arr_tr(
        obj_in = line_in,
        output_type= 'numpy',
        testing = testing
    )

    print('\nOutput: {}\n{} \n'.format(type(numpy_out), numpy_out))



def test_MultiPolygon_to_List_of_numpy(testing = True):
    poly_a = Polygon(((1,2),(3,4),(5,6),(3,2),(1,2)))
    poly_b = Polygon(((0,4),(1,3),(19,1),(2,3),(9,1),(4,1),(0,4)))
    multi_poly = MultiPolygon([poly_a,poly_b])

    print('\nInput: {} \n{} \n'.format(type(multi_poly), multi_poly))

    list_numpy = poly_arr_tr(
        obj_in = multi_poly,
        output_type= 'numpy',
        testing = testing
    )

    print('\nOutput: \n{} \n '.format(list_numpy))


def test_LineString_to_List_of_numpy(testing = True):
    # test
    line_a = LineString(((1,2),(3,4),(5,6),(3,2),(1,2)))
    line_b = LineString(((0,4),(1,3),(19,1),(2,3),(9,1),(4,1),(0,4)))
    multi_lines = MultiLineString([line_a,line_b])

    print('\nInput: {} \n{} \n'.format(type(multi_lines), multi_lines))

    list_numpy = poly_arr_tr(
        obj_in = multi_lines,
        output_type= 'numpy',
        testing = testing
    )

    print('\nOutput: \n{} \n '.format(list_numpy))

def numpy_to_poly(testing = True):
    # test
    numpy_arr = np.array([(1,2),(3,4),(5,6),(3,2),(1,2)])

    print('\nInput: {} \n{} \n'.format(type(numpy_arr), numpy_arr))

    list_poly = poly_arr_tr(
        obj_in = numpy_arr,
        output_type= 'polygon',
        testing = testing
    )

    print('\nOutput: \n{} \n '.format(list_poly))

def numpy_to_line(testing = True):
    # test
    numpy_arr = np.array([(1,2),(3,4),(5,6),(3,2),(1,2)])

    print('\nInput: {} \n{} \n'.format(type(numpy_arr), numpy_arr))

    list_poly = poly_arr_tr(
        obj_in = numpy_arr,
        output_type= 'linestring',
        testing = testing
    )

    print('\nOutput: \n{} \n '.format(list_poly))

def list_numpy_to_multipolygon(testing = True):
    # test
    numpy_arr_a = np.array([(1,2),(3,4),(5,6),(3,2),(1,2)])
    numpy_arr_b = np.array([(0,4),(1,3),(19,1),(2,3),(9,1),(4,1),(0,4)])
    list_numpy = [numpy_arr_a, numpy_arr_b]

    print('\nInput: {} \n{} \n'.format(type(list_numpy), list_numpy))

    list_poly = poly_arr_tr(
        obj_in = list_numpy,
        output_type= 'multipolygon',
        testing = testing
    )

    print('\nOutput: \n{} \n '.format(list_poly))


def list_numpy_to_multilinestring(testing = True):
    # test
    numpy_arr_a = np.array([(1,2),(3,4),(5,6),(3,2),(1,2)])
    numpy_arr_b = np.array([(0,4),(1,3),(19,1),(2,3),(9,1),(4,1),(0,4)])
    list_numpy = [numpy_arr_a, numpy_arr_b]

    print('\nInput: {} \n{} \n'.format(type(list_numpy), list_numpy))

    list_poly = poly_arr_tr(
        obj_in = list_numpy,
        output_type= 'multilinestring',
        testing = testing
    )

    print('\nOutput: \n{} \n '.format(list_poly))


def list_of_lists_of_tuples_TO_list_of_polygons(testing = True):
    numpy_arr_a = [(1,2),(3,4),(5,6),(3,2),(1,2)]
    numpy_arr_b =[(0,4),(1,3),(19,1),(2,3),(9,1),(4,1),(0,4)]
    list_numpy = [numpy_arr_a, numpy_arr_b]

    print('\nInput: {} \n{} \n'.format(type(list_numpy), list_numpy))

    list_poly = poly_arr_tr(
        obj_in = list_numpy,
        output_type= 'multilinestring',
        testing = testing
    )

    print('\nOutput: \n{} \n '.format(list_poly))


def tests_initializer(test_id, testing = True):
    if test_id == 1:
        print('\nTest Polygon_to_Array')
        test_Polygon_to_Array(testing = testing)
        print('Worked\n###########################################')
    elif test_id == 2:
        print('\nTest Polygon_to_Array')
        test_Linestring_to_Array(testing = testing)
        print('Worked\n###########################################')
    elif test_id == 3:
        print('\nTest test_MultiPolygon_to_List_of_numpy')
        test_MultiPolygon_to_List_of_numpy(testing = testing)
        print('Worked\n###########################################')
    elif test_id == 4:
        print('\nTest test_MultiPolygon_to_List_of_numpy')
        test_LineString_to_List_of_numpy(testing = testing)
        print('Worked\n###########################################')
    elif test_id == 5:
        print('\nTest numpy_to_poly')
        numpy_to_poly(testing = testing)
        print('Worked\n###########################################')

    elif test_id == 6:
        print('\nTest numpy_to_line')
        numpy_to_line(testing = testing)
        print('Worked\n###########################################')

    elif test_id == 7:
        print('\nTest list_numpy_to_multipolygon')
        list_numpy_to_multipolygon(testing = testing)
        print('Worked\n###########################################')

    elif test_id == 8:
        print('\nTest list_numpy_to_multilinestring')
        list_numpy_to_multilinestring(testing = testing)
        print('Worked\n###########################################')
    elif test_id == 9:
        print('\nTest list_of_lists_of_tuples_TO_list_of_polygons')
        list_of_lists_of_tuples_TO_list_of_polygons(testing = testing)
        print('Worked\n###########################################')


def call_tests(testing = False):
    test_ids = [1,2,3,4,5,6,7,8]
    for id in test_ids:
        print('\n###########################################')
        tests_initializer(test_id = id, testing = testing)


if __name__ == '__main__':
    call_tests(testing = True)




# Old methods...

#    if type(obj_numpy).__module__ == 'shapely.geometry.polygon':
#        print("")
#        obj_out = np.array(obj_numpy.boundary)
#    elif type(obj_numpy).__module__ == 'numpy':
#        print("")
#        if len(obj_numpy.shape) == 3:
#            if obj_numpy.shape[0] == 1:
#                obj_out = obj_numpy[0,:,:]
#            else:
#                print(obj_numpy)
#                print('hmmm...')
#                print('something is wierd')
#                print('the shape of array (var:obj_numpy) is ', obj_numpy.shape)
#                print('check it out')
#        obj_out = Polygon(list(map(tuple,obj_numpy)))
#    elif type(obj_numpy) == list:
#
#        if type(obj_numpy[0]) == tuple:
#            obj_out = LineString(list(map(tuple,np.array(obj_numpy))))
#        elif type(obj_numpy[0]) == list:
#            contains_sublists = True
#            print('Start While loop')
#            while contains_sublists:
#                temp_obj = []
#                contains_sublists = False
#                for sub_list in obj_numpy:
#                    if type(sub_list[0]) == list:
#                        temp_obj.extend(sub_list)
#                        contains_sublists = True
#                    else:
#                        temp_obj.append(sub_list)
#            print('End While loop')
#            obj_numpy = []
#            for sub_list in temp_obj:
#                obj_numpy.append(LineString(list(map(tuple,np.array(sub_list)))))
#
#            obj_out = MultiLineString(obj_numpy)
#
#    else:                                                               # If obj_in is a MultiPolygon
#        if type(obj_numpy[0]).__module__ == 'shapely.geometry.polygon':       # or obj_in is a list of numpy arrays
#            # Then transform from list of shapely polygons to list of np.ndarrays
#            obj_out = [np.array(xy.boundary) for xy in obj_numpy]
#        elif type(obj_numpy[0]).__module__ == 'numpy':
#            # Then transform from list of np.ndarrays to list of shapely polygons
#            obj_out = MultiPolygon([Polygon(list(map(tuple,arr))) for arr in obj_numpy])
#        else:
#            cow = 'horse'
#            assert cow == 'cow', 'Stop! You used the function "poly_arr_tr" which stands for polygon array transformation. This means it can convert list of polygons to list of array and revers. However, your input is a list of {}'.format(type(obj_in[0]).__module__)
