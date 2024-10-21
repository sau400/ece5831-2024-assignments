import os
import urllib.request
import gzip
import pickle
import numpy as np

class MnistData:
    """OUR MINST DATA CLASS"""  
     
    image_dim = (28, 28)
    image_size = image_dim[0] * image_dim[1]
    dataset_dir = 'dataset'
    dataset_pkl = 'mnist.pkl'

    url_base = 'http://jrkwon.com/data/ece5831/mnist/'  # Alternative: 'https://yann.lecun.com/exdb/mnist/'

    key_file = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def __init__(self):
        self.dataset = {}
        self.dataset_pkl_path = f'{self.dataset_dir}/{self.dataset_pkl}'

        # Create dataset directory if it doesn't exist
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        self._init_dataset()

    def _change_one_hot_label(self, y, num_class):
        t = np.zeros((y.size, num_class))
        for idx, row in enumerate(t):
            row[y[idx]] = 1
        return t

    def _download(self, file_name):
        file_path = os.path.join(self.dataset_dir, file_name)

        if os.path.exists(file_path):
            print(f'File: {file_name} already exists.')
            return

        print(f'Downloading {file_name}...')

        # Resolve 406 Not Acceptable error
        opener = urllib.request.build_opener()
        opener.addheaders = [('Accept', '')]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(self.url_base + file_name, file_path)
        print('Done')

    def _download_all(self):
        for file_name in self.key_file.values():
            self._download(file_name)

    def _load_images(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        return images.reshape(-1, self.image_size)

    def _load_labels(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _create_dataset(self):
        self.dataset['train_images'] = self._load_images(f"{self.dataset_dir}/{self.key_file['train_images']}")
        self.dataset['train_labels'] = self._load_labels(f"{self.dataset_dir}/{self.key_file['train_labels']}")
        self.dataset['test_images'] = self._load_images(f"{self.dataset_dir}/{self.key_file['test_images']}")
        self.dataset['test_labels'] = self._load_labels(f"{self.dataset_dir}/{self.key_file['test_labels']}")

        with open(self.dataset_pkl_path, 'wb') as f:
            print(f'Pickle: {self.dataset_pkl_path} is being created.')
            pickle.dump(self.dataset, f)
            print('Done.')

    def _init_dataset(self):
        self._download_all()
        if os.path.exists(self.dataset_pkl_path):
            with open(self.dataset_pkl_path, 'rb') as f:
                print(f'Pickle: {self.dataset_pkl_path} already exists.')
                print('Loading...')
                self.dataset = pickle.load(f)
                print('Done.')
        else:
            self._create_dataset()

    def load(self):
        # Normalize image datasets
        for key in ('train_images', 'test_images'):
            self.dataset[key] = self.dataset[key].astype(np.float32) / 255.0

        # One-hot encoding of labels
        for key in ('train_labels', 'test_labels'):
            self.dataset[key] = self._change_one_hot_label(self.dataset[key], 10)

        return (self.dataset['train_images'], self.dataset['train_labels']), \
               (self.dataset['test_images'], self.dataset['test_labels'])

# Command-line interface
if __name__ == '__main__':
    print(
        "MnistData class is used to load MNIST datasets.\n"
        "Usage:\n"
        "    load() -> Returns (train_images, train_labels), (test_images, test_labels).\n"
        "    Each image is flattened to 784 bytes. Reshape to display the image.\n"
        "    Each label is one-hot-encoded. Use argmax to get the index of the correct label.\n"
    )
    
