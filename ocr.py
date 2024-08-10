DEBUG = False

if DEBUG: 
    from PIL import Image
    import numpy as np


    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))
    
    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)



DATA_DIR = './data/'

TEST_DIR = './test/'

TEST_DATA_FILENAME = DATA_DIR + 't10k-images-idx3-ubyte'

TEST_LABELS_FILENAME = DATA_DIR + 't10k-labels-idx1-ubyte'

TRAIN_DATA_FILENAME = DATA_DIR + 'train-images-idx3-ubyte'

TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels-idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, byteorder='big')

def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # Magic number
        number_of_images = bytes_to_int(f.read(4))
        if n_max_images:
            number_of_images = n_max_images
        number_of_rows = bytes_to_int(f.read(4))
        number_of_columns = bytes_to_int(f.read(4))
        for image_index in range(number_of_images):
            image = []
            for row_index in range(number_of_rows):
                row = []
                for column_index in range(number_of_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
                    
    return images

def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # Magic number
        number_of_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            number_of_labels = n_max_labels
        for label_index in range(number_of_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
                    
    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist] 

def extract_features(X):
    return [flatten_list(sample) for sample in X]

def dist(x, y):
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i))**2 
            for x_i, y_i in zip(x, y)
        ]
    )**(0.5)

def get_training_distances_for_test_sample(test_sample, X_train):
    return [dist(train_sample, test_sample) for train_sample in X_train]

def get_most_frequent_element(l):
    return max(l, key=l.count)


def accuracy(y_pred, y_test):
    return sum([
        int(ypred_i == ytest_i)
        for ypred_i, ytest_i 
        in zip(y_pred, y_test)
    ]) / len(y_test)


def knn(X_train, y_train, X_test, k=3):
    y_pred_labels = []
    for test_sample_index, test_sample in enumerate(X_test):
        training_distances = get_training_distances_for_test_sample(test_sample, X_train)
        sorted_distance_indices = [
            pair[0] 
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [
            y_train[index]
            for index in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)
        y_pred_labels.append(top_candidate)

    return y_pred_labels

def knn_prediction(drawing, n=4):
    X_train = read_images(TRAIN_DATA_FILENAME, 20000)
    y_train = read_labels(TRAIN_LABELS_FILENAME, 20000) # [0, 1, 5, 9] true labels
    X_train = extract_features(X_train)
    drawing = extract_features(drawing)


    return knn(X_train, y_train, drawing, n)[0]


def find_best_k(X_train, y_train, X_test, y_test):
    best_k = 0
    best_accuracy = 0
    for k in range(1, 10):
        print(f'Checking k={k}')
        y_pred = knn(X_train, y_train, X_test, k)
        current_accuracy = accuracy(y_pred, y_test)
        print(f'k={k}, accuracy={current_accuracy}')
        if current_accuracy > best_accuracy:
            best_k = k
            best_accuracy = current_accuracy
    return best_k, best_accuracy

def find_best_training_size(X_train, y_train, X_test, y_test):
    best_training_size = 0
    best_accuracy = 0
    for training_size in range(1000, 60000, 1000):
        print(f'Checking training_size={training_size}')
        y_pred = knn(X_train[:training_size], y_train[:training_size], X_test, 3)
        current_accuracy = accuracy(y_pred, y_test)
        print(f'training_size={training_size}, accuracy={current_accuracy}')
        if current_accuracy > best_accuracy:
            best_training_size = training_size
            best_accuracy = current_accuracy
    return best_training_size, best_accuracy

def find_best_training_size_and_k(X_train, y_train, X_test, y_test):
    best_training_size = 0
    best_k = 0
    best_accuracy = 0
    for training_size in range(1000, 60000, 1000):
        for k in range(1, 10):
            print(f'Checking training_size={training_size}, k={k}')
            y_pred = knn(X_train[:training_size], y_train[:training_size], X_test, k)
            current_accuracy = accuracy(y_pred, y_test)
            print(f'training_size={training_size}, k={k}, accuracy={current_accuracy}')
            if current_accuracy > best_accuracy:
                best_training_size = training_size
                best_k = k
                best_accuracy = current_accuracy
    return best_training_size, best_k, best_accuracy




def main():
    print("Reading data")
    X_train = read_images(TRAIN_DATA_FILENAME, 60000)
    y_train = read_labels(TRAIN_LABELS_FILENAME, 60000) # [0, 1, 5, 9] true labels
    X_test = read_images(TEST_DATA_FILENAME, 1000)
    y_test = read_labels(TEST_LABELS_FILENAME, 1000) # [0, 1, 5, 9] test labels the ones we want, in this case we know what it is but typically we wouldn't this is just for accuracy measure what we think every image in X_test is
    
    if DEBUG:
        for i, test_sample in enumerate(X_test):
            write_image(test_sample, TEST_DIR + f'test_{i}.png')

    print("Extracting features")
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    print("Starting KNN")
    stand_alone_best_k, stand_alone_best_k_accuracy = find_best_k(X_train, y_train, X_test, y_test)
    stand_alone_best_size, stand_alone_best_size_accuracy= find_best_training_size(X_train, y_train, X_test, y_test)
    overal_best_k, overal_best_size, overal_best_accuracy = find_best_training_size_and_k(X_train, y_train, X_test, y_test)

    print(f'On a test with 60,000 images testing the k, the best k is {stand_alone_best_k} with an accuracy of {stand_alone_best_k_accuracy}')
    print(f'On a test with 3 neighbors, the best training size is {stand_alone_best_size} with an accuracy of {stand_alone_best_size_accuracy}')
    print(f'On a test measuring the highest accuracy, {overal_best_accuracy} can be achieved with a training size of {overal_best_size} and k of {overal_best_k}')
    
    # y_pred = knn(X_train, y_train, X_test, 3)

    # counting_accuracy = accuracy(y_pred, y_test)

    # print(f'Predicted labels: {y_pred}')
    # print(f'Accuracy: {counting_accuracy * 100}%')







            



if __name__ == '__main__':
    main()