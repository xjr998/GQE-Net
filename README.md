# GQE-Net (PyTorch)

## Point Cloud Quality Enhancement
* Run the training script:


``` 1024 points
python main_mix.py --pth_path=pths/final_2023 ----log_path=logs/final_2023 --train_h5_txt=data_LT/h5_mix_new/trainFile.txt --valid_h5_txt=data_LT/h5_mix_test_new/testFile.txt --no_cuda=0 --eval=0 --train_channel=0
```

* Run the evaluation script with pretrained models:

``` 2048 points
python main_mix.py --log_path_test=logs_test_2023/final_2023 --test_ply_txt=data_LT/data_ori_add/testFile.txt --test_ori_ply=data_LT/data_ori_add/same_order/ --test_rec_ply=data_LT/data_rec_add/ --model1_path=pths/final_2023/GQE-Net/2023-07-25/y/model_6.pth --model2_path=pths/final_2023/GQE-Net/2023-07-28/u/model_55.pth ----model3_path=pths/final_2023/GQE-Net/2023-07-31/v/model_92.pth --pred_path=data/preds/final_2023 --eval=1
```

* In the training stage, put the txt file and the .h5 files in the same folder.

