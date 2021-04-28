import numpy as np
import torch
import torch.utils.data as data
import glob
# import tifffile as tiff
# from torchvision import transforms as T

def is_image_file(filename): 
    return any(filename.endswith(extension) for extension in [".tif", '.png', '.jpg', '.npy'])


class train_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0, 
                time_series=4, batch_size=1, transform = None):
        super(train_dataset, self).__init__()
        self.src_list = np.array(sorted(glob.glob(data_path + 'imgs/' + '*.npy')))
        self.lab_list = np.array(sorted(glob.glob(data_path + 'masks/' + '*.npy')))
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        self.time_series = time_series
        self.index = 0
        self.batch_size = batch_size
        self.transform = transform

    def data_iter_index(self, index=1000):
        batch_index = np.random.choice(len(self.src_list), index)
        x_batch = self.src_list[batch_index]
        y_batch = self.lab_list[batch_index]
        data_series = []
        label_series = []

        try:
            for i in range(index):

                # im = tiff.imread(x_batch[i]) / 255.0
                # data_series.append(im[:256, :256, :])
                # mask = tiff.imread(y_batch[i])[:256, :256, :].argmax(axis = -1)
                # label_series.append(mask)
                
                im = np.load(x_batch[i]).transpose([1,2,0]) #/ 255.0  
                # print(data.shape)

                if self.transform:
                  im = self.transform(im)

                data_series.append(im.numpy())
                label_series.append(np.load(y_batch[i]))
                
                self.index += 1

        except OSError:
            return None, None

        
        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        # data_series = data_series.type(torch.FloatTensor)
        # data_series = data_series.permute(0, 3, 1, 2)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True, 
            num_workers=0,
        )

        return data_iter

    def data_iter(self):
        data_series = []
        label_series = []
        try:
            for i in range(len(self.src_list)):

                # im = tiff.imread(self.src_list[i]) / 255.0
                # data_series.append(im[:256, :256, :])
                # mask = tiff.imread(self.lab_list[i])[:256, :256, :].argmax(axis = -1)
                # # label_series.append(rgb_to_1Hlabel(mask).argmax(axis = 0))

                im = np.load(self.src_list[i]).transpose([1,2,0])# / 255.0    
                   
                if self.transform:
                  im = self.transform(im)

                data_series.append(im.numpy())
                label_series.append(np.load(self.lab_list[i]))

                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        # data_series = data_series.permute(0, 3, 1, 2)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True, 
            num_workers=0, 
        )

        return data_iter