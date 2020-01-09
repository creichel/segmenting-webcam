import segmentation_models as sm
import utils
import keras
import dataset
import augmentation
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BACKBONE = 'mobilenetv2'
BATCH_SIZE = 8
CLASSES = ['car', 'pedestrian']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5]))
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# Dataset for train images
train_dataset = dataset.Dataset(
    dataset.x_train_dir,
    dataset.y_train_dir,
    classes=CLASSES,
    augmentation=augmentation.get_training_augmentation(),
    preprocessing=augmentation.get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = dataset.Dataset(
    dataset.x_valid_dir,
    dataset.y_valid_dir,
    classes=CLASSES,
    augmentation=augmentation.get_validation_augmentation(),
    preprocessing=augmentation.get_preprocessing(preprocess_input),
)

train_dataloader = dataset.Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = dataset.Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

# train model
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)

test_dataset = dataset.Dataset(
    dataset.x_test_dir,
    dataset.y_test_dir,
    classes=CLASSES,
    augmentation=augmentation.get_validation_augmentation(),
    preprocessing=augmentation.get_preprocessing(preprocess_input),
)

test_dataloader = dataset.Dataloder(test_dataset, batch_size=1, shuffle=False)

# load best weights
model.load_weights('best_model-multiclass.h5')

scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n)

for i in ids:
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)

    utils.visualize(
        image=utils.denormalize(image.squeeze()),
        gt_mask=gt_mask.squeeze(),
        pr_mask=pr_mask.squeeze(),
    )