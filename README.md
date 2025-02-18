## GAP-CLIP: AreYouFocused?EmpoweredCLIPWithGAPFeaturesForStudentEngagementDetection

### Result

| **Method**             | **Input**           | **BackBone** | **DAiSEE** | **EngageNet** | **Avg**   |
| ---------------------- | ------------------- | ------------ | ---------- | ------------- | --------- |
| C3D+TCN                | frames              | C3D          | 54.09      | 55.17         | 54.63     |
| InceptionNet           | frames              | InceptionNet | 50.90      | 53.30         | 51.70     |
| EfficientNetB7+LSTM    | frames              | EfficientNet | 52.58      | 56.44         | 54.51     |
| EfficientNetB7+Bi-LSTM | frames              | EfficientNet | 54.73      | 56.54         | 55.64     |
| Resnet+TCN             | frames              | Resnet       | 55.72      | 57.03         | _56.38_   |
| CLIP                   | frames              | CLIP         | 48.35      | 52.64         | 50.40     |
| CMOSE                  | GAP features+audio  | I3D          | 47.70      | -             | -         |
| TCCNET($aug$)          | GAP features        | Transformer  | 50.38      | **59.78**     | 55.08     |
| **ours**               | frames+GAP features | CLIP         | **57.86**  | _59.63_       | **58.75** |

### Requirement

The code is built with following libraries: **PyTorch, scikit-learn, einops, matplotlib, numpy, math, shutil, tqdm, python=3.10**

### Run the train code

+ download the dataset [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) and [EngageNet](https://github.com/engagenet/engagenet_baselines)

  ```
  mkdir data
  mkdir pretrain
  ```

+ use the *dataloader.VideoProcess.py* to process the datasets

  After this, there will be label files in the annotation folder and data files in data folder.

+ download the ckpt of [clip-Vit-B-32](https://huggingface.co/sentence-transformers/clip-ViT-B-32) and place in the pretrain foloder

+ use the *train_DAiSEE.sh* or *train_EmotiW.sh* to run the train code

### Run baselines

We have downloaded or reconstructed all baselines used in our paper

**models.GenerateModel.py** : EfficientNet-LSTM, EfficientNet-BiLstm, InceptionNet

**models.ResnetTCN**: ResNet+TCN

**TCCNet**: We modified the TCCNet code to train the model use the train, valid and test sets instead of mixing the train and valida sets.

If you want to run the baselines you can modify the main.py and uncomment the corresponding sections in **main.py**.

For example, if you want to run the **EfficientNet-LSTM**:

```python
    model = EfficentLstm()
    ...
    for name, param in model.named_parameters():
        if "lstm" in name:
            param.requires_grad = True  
            if "out_fc" in name:
                param.requires_grad = True
                if "conformer" in name:  
                    param.requires_grad = True
    ...
    if "classification_head_conformer" in name:
    param.requires_grad = True
    optimizer = torch.optim.SGD([{"params": model.module.lstm.parameters(), "lr": args.lr_temporal_net},
                            {"params": model.module.out_fc.parameters(), "lr": args.lr_temporal_net},
                            # {"params": model.module.conformer.parameters(), "lr": args.lr_temporal_net},
                            # {"params": model.module.classification_head_conformer.parameters(), "lr": args.lr_temporal_net}
                            ],
                            lr=args.lr_temporal_net,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
      ...
```

