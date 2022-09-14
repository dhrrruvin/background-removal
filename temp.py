# from data import process_data

# image, mask = process_data("dataset/train/", "images", "labels")

# print(image)
# print("-----------------------------------------------------")
# print(mask)
import cv2 as cv
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# def adjust_data(img,mask,flag_multi_class,num_class):
#     if(flag_multi_class):
#         img = img / 255
#         mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
#         new_mask = np.zeros(mask.shape + (num_class,))
#         for i in range(num_class):
#             #for one pixel in the image, find the class in mask and convert it into one-hot vector
#             #index = np.where(mask == i)
#             #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
#             #new_mask[index_mask] = 1
#             new_mask[mask == i,i] = 1
#         new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
#         mask = new_mask
#     elif(np.max(img) > 1):
#         img = img / 255
#         mask = mask /255
#         mask[mask > 0.5] = 1
#         mask[mask <= 0.5] = 0
#     return (img,mask)

def train_generator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (1026,1225),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        # img,mask = adjust_data(img,mask,flag_multi_class,num_class)
        yield (img,mask)

if __name__ == "__main__":
    # data_gen_args = dict(rotation_range=0.2,
    #                 width_shift_range=0.05,
    #                 height_shift_range=0.05,
    #                 shear_range=0.05,
    #                 zoom_range=0.05,
    #                 horizontal_flip=True,
    #                 fill_mode='nearest')
    # myGenerator = train_generator(20, 'dataset/train', 'images', 'masks', data_gen_args, save_to_dir = "dataset/train/aug")

    # num_batch = 3
    # for i,batch in enumerate(myGenerator):
    #     if(i >= num_batch):
    #         break

    files = os.listdir("dataset/train/aug")

    for file in files:
        if file[0] == 'm':
            img = cv.imread('dataset/train/aug/mask_0_2104065.png', cv.IMREAD_COLOR)
            # img = cv.subtract(255, img)

            w = np.where(img[:, :] == 255)
            b = np.where(img[:, :] == 0)

            img[w] = 0
            img[b] = 255

            cv.imwrite(os.path.join("dataset/train/masks", file), img)
    #         img = cv.imread(file, cv.IMREAD_COLOR)
    #         cv.imwrite(file, )