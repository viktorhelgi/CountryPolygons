import matplotlib.pyplot as plt
from CoordinateObjects import PolyCoordsCollection



import pickle



def main():
    file = 'data/pickles/denmark_PolyCoords.p'
    with open(file, 'rb') as handle:
        data:PolyCoordsCollection = pickle.load(handle)


if __name__ == '__main__':
    main()
