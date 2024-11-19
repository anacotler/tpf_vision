#split data into 3 csv: train, validation, test

from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data, train_size, validation_size, test_size):
    #split data into train, validation, test
    dev, test = train_test_split(data, test_size = test_size)
    train, validation = train_test_split(dev, test_size = validation_size)
    return train, validation, test

def main():
    data = pd.read_csv('utkface_dataset.csv')
    train, validation, test = split_data(data, 0.7, 0.15, 0.15)
    train.to_csv('train.csv', index=False)
    validation.to_csv('validation.csv', index=False)
    test.to_csv('test.csv', index=False)

if __name__ == '__main__':
    main()
    

