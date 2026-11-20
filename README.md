# Insect Classifier: Identifying Species from Photos with Transfer Learning

Can a computer tell three insect species apart from a photograph, and how little training does it take? This project fine-tunes a pre-trained **ResNet-18** on a small insect image set and uses the result to label new pictures with a confidence score.

## The data

Photos of **three insect species** (one folder per species), split into **686 training**, **135 validation** and **95 test** images (75 / 15 / 10 %) with a fixed random seed so the split is reproducible.

## The approach

Training a network from scratch on fewer than 700 images would overfit badly. Instead the model **starts from ResNet-18 pre-trained on ImageNet**, whose layers already detect edges, textures and shapes, and only the final layer is replaced by a new 3-way classifier. The whole network is then fine-tuned for 10 epochs.

- **Augmentation** keeps the model from memorising: random rotations, random resized crops to 224 × 224 and horizontal flips on the training images; plain resize and centre crop for validation.
- **Normalisation** uses the ImageNet mean and standard deviation, which the pre-trained weights expect.
- **Optimisation** is SGD with momentum and a step schedule that cuts the learning rate by a factor of 10 every 7 epochs: large steps early, fine adjustments at the end.

## What the results show

| | Accuracy |
|---|---|
| Training (with augmentation) | ≈ 94.6 % |
| Validation | **100 %** |

Training took about 12 minutes. Validation accuracy being *higher* than training accuracy is normal here, because the training images are heavily augmented (rotated and cropped) while the validation images are not, so the training set is the harder one. With three visually distinct species and only 135 validation images, a perfect score also says the task is fairly easy rather than that the model is flawless; the next test would be harder species or photos taken under different conditions. In the final step the trained model labels new photos and prints the predicted species with a confidence score, which is close to 1.00 on the examples shown.

## What you can take from this

1. Transfer learning turns a few hundred images into a working classifier in minutes.
2. Augmentation is cheap insurance against overfitting on small datasets, and it explains why training accuracy can lag behind validation accuracy.
3. A confidence score next to each prediction tells you how much to trust it, which becomes essential once the model meets species or conditions it was not trained on.

## Files

- `main.ipynb`: data split, augmentation, fine-tuning, evaluation and inference
- `database/`: the images, one folder per species
- `models/`: the trained weights (`model.pth`) and the class names (`model.pkl`)
