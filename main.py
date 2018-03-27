from nn import NeuralNetwork
import argparse
import numpy as np
# from PIL import Image
import re


parser = argparse.ArgumentParser()
parser.add_argument("-V", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()
verbose = False

if args.verbose:
    verbose = True


def read_pgm(filename,  plot=False, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    Following code is inspired from https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm

    """
    if verbose:
        print('reading image', filename)
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        # print (header, int(width), int(height), int(maxval))
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    image = np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ) #.reshape((int(height), int(width)))
    # print(image)
    # to save image
    if plot:
        im = Image.fromarray(image, 'L')
        im.putdata(image)
        im.save(filename + '.jpg')
        im.show()
    return image


def fetch_data(filename='downgesture_train.list'):
    if verbose:
        print('Reading ', filename)
    data = []
    labels = []
    # files = []
    with open(filename) as f:
        for line in f:
            if line:
                # data.append(line.strip())
                # print(line.strip())
                line = line.strip()
                # files.append(line)
                data.append(read_pgm(line))
                if "down" in line:
                    labels.append(1)
                else:
                    labels.append(0)
    if verbose:
        # print(data)
        # print(files)
        print(labels)
        print()

    return data, labels


if __name__ == '__main__':
    train_data, train_target = fetch_data('downgesture_train.list')
    neuralnet = NeuralNetwork()
    neuralnet.add_layer(size = 1, input_size = len(train_data[0]), type='input')
    neuralnet.add_layer(size = 100)
    neuralnet.add_layer(size = 1, type='output')
    
    neuralnet.fit(data = train_data, target = train_target, eta = 0.1, verbose = verbose)
    # if needed clean data

    # fit a model
    # train a model
    test_data, test_target = fetch_data('downgesture_test.list')
    predicted_target = neuralnet.predict(test_data)
    accuracy = neuralnet.accuracy(test_target, predicted_target)
    print("Accuracy:", accuracy)
