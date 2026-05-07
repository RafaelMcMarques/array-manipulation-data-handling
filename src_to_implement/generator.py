import os.path
import json
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        self.file_path = file_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        with open(label_path, "r") as f:
            self.label_path = json.load(f)

        self.indices = list(range(len(self.label_path)))
        self.current_image_index = 0

        self.current_epoch_num = 0

        self.epoch_finished = False
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)




    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        images = []
        labels = []

        if self.epoch_finished:
            self.current_epoch_num += 1
            self.epoch_finished = False
            if self.shuffle:
                np.random.shuffle(self.indices)

        for i in range(self.batch_size):
            idx = self.indices[self.current_image_index]
            image_path = os.path.join(self.file_path, f'{idx}.npy')
            image = np.load(image_path)

            ## resize ## 
            h, w, c = image.shape
            new_h, new_w, new_c = self.image_size
            row_idx = np.linspace(0, h - 1, new_h).astype(int)
            col_idx = np.linspace(0, w - 1, new_w).astype(int)
            image = image[row_idx][:, col_idx]

            if self.rotation or self.mirroring:
                image = self.augment(image)
            images.append(image)
            label = int(self.label_path[str(idx)])
            labels.append(label)
            self.current_image_index += 1
            if self.current_image_index == len(self.indices):
                self.current_image_index = 0
                self.epoch_finished = True
                
        #return images, labels
        return np.array(images), np.array(labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        choice = np.random.choice([0, 1, 2])
        if choice == 0:
            if self.rotation:
                angle = np.random.choice([1, 2, 3])
                img = np.rot90(img, angle) # rotate the image by the chosen angle
        if choice == 1:
            if self.mirroring:
                img = np.fliplr(img) # flip the image horizontally
        if choice == 2:
            pass # do nothing
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.current_epoch_num

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        batch = self.next(resize=True)
        fig, axes = plt.subplots(1, self.batch_size, figsize=(15, 5))
        for i in range(self.batch_size):
            axes[i].imshow(batch[0][i])
            axes[i].set_title(batch[1][i])
            axes[i].axis('off')
        plt.show()
