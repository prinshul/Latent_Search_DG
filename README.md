# Domain Generalization with Latent Search

## Setup

```bash
git clone https://github.com/prinshul/Latent_Search_DG.git
cd Latent_Search_DG
conda create -y -n ltp python=3.10
conda activate ltp
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 pillow==10.4.0
```


```bash
cd pacs_resnet18
bash run.sh
```

```bash
cd pacs_resnet50
bash run.sh --arch resnet50 --select_lr
```

```bash
cd officehome_resnet18
Download dataset from https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw
unzip OfficeHomeDataset_10072016.zip
DATA=OfficeHomeDataset_10072016 bash run.sh
```
