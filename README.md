# Implementation of our CVPR 2019 paper Distilling Object Detectors with Fine-grained Feature Imitation

![15\% performance boost of student model](https://github.com/twangnh/Distilling-Object-Detectors/blob/master/intuition.PNG)

We have proposed a general distillation approach for anchor based object detection model to get enhanced small student model with the knowledge of large teacher model, which is othorgonal and can be further combined with other model compression method like quantization and pruning. The key observation of vanilla knowledge distillation technique is that the inter-class discrepancy of perdiction confidence reveals how curmbersome model tends to genearlize (e.g., how much confidence the model would put on cat label when the input is actually a dog). While our idea is the inter-location discrepancy of feature response near object also reveals how large detector tends to generalize (e.g., how is the model's response different for different near object anchor locations).

We release the code for distilling shufflenet based detector and **VGG11** based **Faster R-CNN**, this code repository implements **Faster R-CNN** imitation based on [pytorch-faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch). Check [Distilling-ShuffleDet](https://github.com/twangnh/Distilling-Object-Detectors-Shuffledet) for tensorflow code of Shufflenet based detector imitation.

🔥Updating🔥
## TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [ ] Add more models(ResNet-FRCNN, FPN-FRCNN).
  * [ ] Implement SSD model distillation.

## Distilling VGG11-FRCNN
pytorch 0.4.0 python2

### Preparation

#### 1 Clone the repository
First of all, clone the code
```
git clone https://github.com/twangnh/Distilling-Object-Detectors
```

Then, create a folder:
```
cd Distilling-Object-Detectors && mkdir data
```
#### 2 Data preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, create softlinks in the folder data/. The prepaired direcoty is like data/VOCdevkit2007/VOC2007/...

#### 3 download imagenet pretrained model and trained VGG16-FRCNN teacher model
download imagenet pretrained VGG11 model at [GoogleDrive](https://drive.google.com/file/d/13Kvewg9FFg4EgIpRZeZPtIUllDiaE7aY/view?usp=sharing) and put it into data/pretrained_model/

download trained VGG16-FRCNN model at [GoogleDrive](https://drive.google.com/file/d/1RB8wP0Pf7bMv_iArYomVOfq1ikTsaA2O/view?usp=sharing)
and put it into data/VGG16-FRCNN/


### Train
currently only batch size of 1 is supported
```
python trainval_net_sup.py --dataset pascal_voc --net vgg11 --bs 1 --nw 2 --lr 3e-3 --lr_decay_step 5 --cuda --s 1 --gpu 0
```

```
[session 1][epoch  1][iter    0/10022] loss: 13.4238, loss_sup: 0.0000, lr: 3.00e-03
			fg/bg=(15/241), time cost: 0.307381
			rpn_cls: 0.7839, rpn_box: 0.4312, rcnn_cls: 12.0130, rcnn_box 0.1957 
[session 1][epoch  1][iter  100/10022] loss: 2.1172, loss_sup: 0.0000, lr: 3.00e-03
			fg/bg=(15/241), time cost: 17.871297
			rpn_cls: 0.2372, rpn_box: 0.0492, rcnn_cls: 0.5382, rcnn_box 0.1475 
[session 1][epoch  1][iter  200/10022] loss: 2.3993, loss_sup: 0.0000, lr: 3.00e-03
			fg/bg=(27/229), time cost: 17.885193
			rpn_cls: 0.0451, rpn_box: 0.3003, rcnn_cls: 2.5547, rcnn_box 0.5216 
[session 1][epoch  1][iter  300/10022] loss: 1.6754, loss_sup: 0.0000, lr: 3.00e-03
			fg/bg=(21/235), time cost: 17.856990
			rpn_cls: 0.2837, rpn_box: 0.2542, rcnn_cls: 1.1131, rcnn_box 0.2073 
[session 1][epoch  1][iter  400/10022] loss: 1.6178, loss_sup: 0.1145, lr: 3.00e-03
			fg/bg=(23/233), time cost: 17.976755
			rpn_cls: 0.3597, rpn_box: 0.0106, rcnn_cls: 0.7343, rcnn_box 0.2363 
[session 1][epoch  1][iter  500/10022] loss: 1.4362, loss_sup: 9.6434, lr: 3.00e-03
			fg/bg=(32/224), time cost: 17.911143
			rpn_cls: 0.1783, rpn_box: 0.0235, rcnn_cls: 0.4522, rcnn_box 0.3731 
[session 1][epoch  1][iter  600/10022] loss: 1.3638, loss_sup: 8.4568, lr: 3.00e-03
			fg/bg=(18/238), time cost: 18.024369
			rpn_cls: 0.4774, rpn_box: 0.1143, rcnn_cls: 0.4781, rcnn_box 0.1663 
```
>training should progress as above, where loss_sup is the imitation loss and firts 400 steps are warmup steps with no imitation (i.e., loss_sup = 0.)
>models will be saved in ```./temp/vgg11/pascal_voc/xxx.pth```

### Train without imitation (baseline training)
currently only batch size of 1 is supported
```
python trainval_net_sup.py --dataset pascal_voc --net vgg11 --bs 1 --nw 2 --lr 3e-3 --lr_decay_step 5 --cuda --s 1 --gpu 0 --tfi True
```
>models will be saved in ```./temp/vgg11/pascal_voc/xxx.pth```
>Note the imitation loss weight and warm up step can be further tuned with --ilw and --ws

### Test
```
python test_net.py --dataset pascal_voc --net vgg11 --checksession 1 --checkepoch 2 --checkpoint 10021 --cuda --gpu 0
```
change ```checksession```, ```checkepoch```, ```checkpoint``` to test specific model

###
model    | #GPUs | batch size | learning_rate(lr) | lr_decay | max_epoch | mAP | ckpt
-------------|--------|---------|--------|-----|------|-----|----
VGG-16     | 1 | 1 | 1e-3 | 5   | 7   | 70.1 | [GoogleDrive](https://drive.google.com/file/d/1RB8wP0Pf7bMv_iArYomVOfq1ikTsaA2O/view?usp=sharing)
VGG-11     | 1 | 4 | 3e-3 | 8   | 15   | 59.6 | [GoogleDrive](https://drive.google.com/file/d/1dEsOcVxOmm8KTw3J3EwhPXeSABEP1TuN/view?usp=sharing)
VGG-11-I   | 8 | 16| 3e-3 | 8   | 15   | 67.6 **+8.0** | [GoogleDrive](https://drive.google.com/file/d/1mIzuAv_cMTvaSEVRCF8OhS11Em7b-onB/view?usp=sharing)
>models at max_epoch are reported

>the numbers are different from the paper as they are independent running of the algorithm.

### Test with trained model
download the trained model at the GoogleDrive link, run
```
python test_net.py --dataset pascal_voc --net vgg11 --load_name ./path_to/xxx.pth --cuda --gpu 0
```

## Distilling ShuffleDet
...

We have implemented a single layer one-stage toy object detector with tensorflow, and mutli-gpu training with cross-gpu batch normalization, check [Distilling-ShuffleDet](https://github.com/twangnh/Distilling-Object-Detectors-Shuffledet) for codes

<table class="tg">
  <tr>
    <th class="tg-k19b" rowspan="2">Models</th>
    <th class="tg-k19b" rowspan="2">Flops<br>/G</th>
    <th class="tg-gom2" rowspan="2">Params<br>/M</th>
    <th class="tg-gom2" colspan="3">car</th>
    <th class="tg-gom2" colspan="3">pedestrian</th>
    <th class="tg-gom2" colspan="3">cyclist</th>
    <th class="tg-gom2" rowspan="2">mAP</th>
    <th class="tg-gom2" rowspan="2">ckpt</th>
  </tr>
  <tr>
    <td class="tg-gom2">Easy</td>
    <td class="tg-gom2">Mod</td>
    <td class="tg-gom2">Hard</td>
    <td class="tg-gom2">Easy</td>
    <td class="tg-gom2">Mod</td>
    <td class="tg-gom2">Hard</td>
    <td class="tg-gom2">Easy</td>
    <td class="tg-gom2">Mod</td>
    <td class="tg-gom2">Hard</td>
  </tr>
  <tr>
    <td class="tg-k19b">1x</td>
    <td class="tg-k19b">5.1</td>
    <td class="tg-gom2">1.6</td>
    <td class="tg-gom2">85.7</td>
    <td class="tg-gom2">74.3</td>
    <td class="tg-gom2">65.8</td>
    <td class="tg-gom2">63.2</td>
    <td class="tg-gom2">55.6</td>
    <td class="tg-gom2">50.6</td>
    <td class="tg-gom2">69.7</td>
    <td class="tg-gom2">51.0</td>
    <td class="tg-gom2">49.1</td>
    <td class="tg-tzpo">62.8</td>
    <td class="tg-gom2"><a href="https://drive.google.com/file/d/1jTTeyA61ZPDgwrvHeelysTQLUSWbIOul/view?usp=sharing">GoogleDrive</a></td>
  </tr>
  <tr>
    <td class="tg-k19b">0.5x</td>
    <td class="tg-k19b">1.5</td>
    <td class="tg-gom2">0.53</td>
    <td class="tg-gom2">81.6</td>
    <td class="tg-gom2">71.7</td>
    <td class="tg-gom2">61.2</td>
    <td class="tg-gom2">59.4</td>
    <td class="tg-gom2">52.3</td>
    <td class="tg-gom2">45.5</td>
    <td class="tg-gom2">59.7</td>
    <td class="tg-gom2">43.5</td>
    <td class="tg-gom2">42.0</td>
    <td class="tg-tzpo">57.4</td>
    <td class="tg-gom2"><a href="https://drive.google.com/file/d/1SYWeyRLbeDIQZzETKy0oqONe5hvepWqK/view?usp=sharing">GoogleDrive</a></td>
  </tr>
  <tr>
    <td class="tg-k19b" rowspan="2">0.5x-I</td>
    <td class="tg-k19b" rowspan="2">1.5</td>
    <td class="tg-gom2" rowspan="2">0.53</td>
    <td class="tg-gom2">84.9</td>
    <td class="tg-gom2">72.9</td>
    <td class="tg-gom2">64.1</td>
    <td class="tg-gom2">60.7</td>
    <td class="tg-gom2">53.3</td>
    <td class="tg-gom2">47.2</td>
    <td class="tg-gom2">69.0</td>
    <td class="tg-gom2">46.2</td>
    <td class="tg-gom2">44.9</td>
    <td class="tg-tzpo">60.4</td>
    <td class="tg-gom2"><a href="https://drive.google.com/file/d/1TyU7b957pRkD5PGHgoPy6803XAK0oTK3/view?usp=sharing">GoogleDrive</a></td>
  </tr>
  <tr>
    <td class="tg-oesp">+3.3</td>
    <td class="tg-oesp">+1.2</td>
    <td class="tg-oesp">+2.9</td>
    <td class="tg-oesp">+1.3</td>
    <td class="tg-oesp">+1.0</td>
    <td class="tg-oesp">+1.7</td>
    <td class="tg-oesp">+9.3</td>
    <td class="tg-oesp">+2.7</td>
    <td class="tg-oesp">+2.9</td>
    <td class="tg-oesp">+3.0</td>
    <td class="tg-186s"></td>
  </tr>
  <tr>
    <td class="tg-k19b">0.25x</td>
    <td class="tg-k19b">0.67</td>
    <td class="tg-gom2">0.21</td>
    <td class="tg-gom2">67.2</td>
    <td class="tg-gom2">56.6</td>
    <td class="tg-gom2">47.5</td>
    <td class="tg-gom2">54.7</td>
    <td class="tg-gom2">48.4</td>
    <td class="tg-gom2">42.1</td>
    <td class="tg-gom2">49.1</td>
    <td class="tg-gom2">33.3</td>
    <td class="tg-gom2">32.9</td>
    <td class="tg-gom2">48.0</td>
    <td class="tg-gom2"><a href="https://drive.google.com/file/d/1NLTUI7nJx6-6BjncxUx0nJ9vX5FqcS_C/view?usp=sharing">GoogleDrive</a></td>
  </tr>
  <tr>
    <td class="tg-k19b" rowspan="2">0.25x-I</td>
    <td class="tg-k19b" rowspan="2">0.67</td>
    <td class="tg-gom2" rowspan="2">0.21</td>
    <td class="tg-gom2">76.6</td>
    <td class="tg-gom2">62.3</td>
    <td class="tg-gom2">54.6</td>
    <td class="tg-gom2">56.8</td>
    <td class="tg-gom2">48.2</td>
    <td class="tg-gom2">42.6</td>
    <td class="tg-gom2">56.6</td>
    <td class="tg-gom2">37.3</td>
    <td class="tg-gom2">36.5</td>
    <td class="tg-gom2">52.4</td>
    <td class="tg-gom2"><a href="https://drive.google.com/file/d/18J350FX3MSTG5pgvXDXPfuvx0xxbDl1V/view?usp=sharing">GoogleDrive</a></td>
  </tr>
  <tr>
    <td class="tg-186s">+9.4</td>
    <td class="tg-186s">+5.7</td>
    <td class="tg-186s">+7.1</td>
    <td class="tg-186s">+2.1</td>
    <td class="tg-186s">-0.2</td>
    <td class="tg-186s">+0.5</td>
    <td class="tg-186s">+7.5</td>
    <td class="tg-186s">+4.0</td>
    <td class="tg-186s">+3.6</td>
    <td class="tg-186s">+4.4</td>
    <td class="tg-186s"></td>
  </tr>
</table>

>models with highest mAP are reported for both baseline and distilled model

>**Note** the numbers are different from the paper as they are independent running of the algorithm and we have migrated from single GPU training to multi-gpu training with larger batch size.

## Distilling YoloV2

Third party implementation of distilling YOLOV2 on Widerface(codes not available yet, but very easy to implement)

Model   | `size` | easy | medium | hard |
:-------|:----:|:----|:----|:----
YOLOv2  | `190MB` | 87.2 | 74.6 | 36.0 |
0.25x   | `12MB`  | 78.2 | 69.8 | 35.6 |
0.25x-I | `12MB`  | 83.9 **+5.7** | 74.9 **+5.1** | 38.5 **+2.9** |
0.15x   | `4.4MB` | 69.7 | 61.1 | 29.7 |
0.15x-I | `4.4MB` | 79.3 **+9.6** | 67.0 **+5.9** | 32.0 **+2.3** |

## Citation
```
@inproceedings{wang2019distilling,
  title={Distilling Object Detectors With Fine-Grained Feature Imitation},
  author={Wang, Tao and Yuan, Li and Zhang, Xiaopeng and Feng, Jiashi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4933--4942},
  year={2019}
}
```
## License
The code and the models are MIT licensed, as found in the LICENSE file.
