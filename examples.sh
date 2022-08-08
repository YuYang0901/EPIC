### From-Scratch
```
- GM + CIFAR10 + ResNet-18 (subset_size=0.1/0.2/0.3)
- eps=8, epochs=40
```
python train_poison_epic.py --gpu-id $GPU_ID$ --arch resnet18 --poisons_path $POISONS_PATH$ --seed $SEED$ --out $OUTPUT_PATH$ --epoch 40 --subset_size $SUBSET_SIZE$ --subset_freq 2 --drop_after 10 

```
- eps=8, epochs=200
```
python train_poison_epic.py --gpu-id $GPU_ID$ --arch resnet18 --poisons_path $POISONS_PATH$ --seed $SEED$ --out $OUTPUT_PATH$ --epoch 200 --subset_size $SUBSET_SIZE$ --subset_freq 10 --drop_after 30 
 
```
- eps=16, epochs=40
```
python train_poison_epic.py --gpu-id $GPU_ID$ --arch resnet18 --poisons_path $POISONS_PATH$ --seed $SEED$ --out $OUTPUT_PATH$ --epoch 40 --subset_size $SUBSET_SIZE$ --subset_freq 5  


```
- BP + TinyImageNet + VGG-16
```
python train_poison_epic.py --gpu-id $GPU_ID$ --arch vgg16 --dataset tinyimagenet --poisons_path $POISONS_PATH$ --seed $SEED$ --out $OUTPUT_PATH$ --epoch 200 --subset_size 0.2 --subset_freq 10

### Transfer
```
- BP
```
python train_poison_epic.py --gpu-id $GPU_ID$ --arch resnet18 --poisons_path $POISONS_PATH$ --seed $SEED$ --out $OUTPUT_PATH$ --subset_size 0.2 --subset_freq 1 --transfer

```
- FC
```
python train_poison_epic.py --gpu-id $GPU_ID$ --arch resnet18 --poisons_path $POISONS_PATH$ --seed $SEED$ --out $OUTPUT_PATH$ --subset_size 0.2 --subset_freq 1 --transfer


### Fine-Tuning
```
- BP
```
python train_poison_epic.py --gpu-id $GPU_ID$ --arch resnet18 --poisons_path $POISONS_PATH$ --seed $SEED$ --out $OUTPUT_PATH$ --scenario finetune --subset_size 0.6 --subset_freq 1 --ap_model --drop_after 5
