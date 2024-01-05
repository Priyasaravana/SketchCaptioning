import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import xarray as xr
import s3fs


class CaptionDatasetTest(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        
        # Load encoded captions (completely into memory)
        with open('image_data.json', 'r') as json_file:
            image_data = json.load(json_file)

        # Now, image_data is a list of dictionaries, each containing "image_name" and "caption"
        # You can iterate through the list to access each element
        for item in image_data:
            self.image_path = item["image_path"]
            self.captions = item["caption"]       

        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        
        #caption = torch.LongTensor(self.captions[i])
        caption = self.captions[i]    
        image_path = self.image_path[i]


        # return img, caption, caplen
        # if self.split is 'TRAIN':
        return image_path, caption      

    def __len__(self):        
        return self.dataset_size
