# Localizing-Visual-Sounds-the-Hard-Way

Code and Dataset for "[**Localizing Visual Sounds the Hard Way**](https://arxiv.org/pdf/2104.02691.pdf)".

The repo contains code and our pre-trained model. 

## Environment

* Python 3.6.8
* Pytorch 1.3.0

## Flickr-SoundNet

We provide the pretrained [**model**](https://drive.google.com/file/d/1Rif19nk8RpxBrkGGoI92g-fssa2qsl-m/view?usp=sharing) here.

To test the model, testing data and ground truth should be downloaded from [**learning to localize sound source**](https://github.com/ardasnck/learning_to_localize_sound_source).

Then run

```
python test.py --data_path "path to downloaded data with structure below/" --summaries_dir "path to pretrained models" --gt_path "path to ground truth" --testset "flickr"
```


## VGG-Sound Source

We provide the pretrained [**model**](https://drive.google.com/file/d/1ieyJ7rrrR5aF8SwHNGYNS9dX9t9_apsc/view?usp=sharing) here.

To test the model, run

```
python test.py --data_path "path to downloaded data with structure below/" --summaries_dir "path to pretrained models" --testset "vggss"
```

(Note, some gt bounding boxes are updated recently, all results on VGG-SS cause a 2~3% difference on IoU.)

Both test data should be placed in the following structure.

```
data path
│
└───frames
│   │   image001.jpg
│   │   image002.jpg
│   │
└───audio
    │   audio011.wav
    │   audio012.wav
```




## Citation
```
@InProceedings{Chen21,
              title        = "Localizing Visual Sounds the Hard Way",
              author       = "Honglie Chen, Weidi Xie, Triantafyllos Afouras, Arsha Nagrani, Andrea Vedaldi, Andrew Zisserman",
              booktitle    = "CVPR",
              year         = "2021"}
```