from torch.utils.data import Dataset

query = 'SELECT * FROM {};'


class TrainingData(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = cursor.execute(query.format('train'))

    def __getitem__(self, index):
        return self.data[index]
