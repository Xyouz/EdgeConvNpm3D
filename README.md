# Segmentation of the Stanford indoor dataset

&nbsp;
## Requirements
- Windows OS (A few command changes to make it works on Unix)
- Python 3.7
- PyTorch 1.2
- CUDA 10.0
- Package: glob, h5py, sklearn



## Point Cloud Sementic Segmentation

You have to download `Stanford3dDataset_v1.2_Aligned_Version.zip` manually from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under `data/`

### Run the training script:

- Train in area 1-5

```
python main_seg.py --exp_name=semseg --test_area=6
```

### Run the evaluation script after training finished:

- Evaluate with the trained model :

```
python main_seg.py --exp_name=semseg_eval_6 --test_area=6 --eval=True --model_root=checkpoints/semseg/models/
```


### Run the evaluation script with our tho model we trained:

- Evaluate in area 6 with our pretrained model

```
python main_seg.py --exp_name=semseg_eval_6 --test_area=6 --eval=True --model_root=pretrained/semseg/
```
