import dataset
import utils
import augmentation

# Lets look at data we have
dataset = dataset.Dataset(dataset.x_train_dir, dataset.y_train_dir, classes=['car', 'pedestrian'], augmentation=augmentation.get_training_augmentation())

image, mask = dataset[12]  # get some sample
utils.visualize(
    image=image,
    cars_mask=mask[..., 0].squeeze(),
    sky_mask=mask[..., 1].squeeze(),
    background_mask=mask[..., 2].squeeze(),
)
