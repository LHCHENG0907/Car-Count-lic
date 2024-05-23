import os
import json
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from PIL import Image

# 選定數據集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, transforms=None):
        self.image_folder = image_folder
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        
        # print出前五行做確認
        print("Annotations:", self.annotations[:5])
        
        self.transforms = transforms

        # 檢測每個項目是否都有image
        for i, annotation in enumerate(self.annotations):
            if 'image' not in annotation:
                raise KeyError(f"'image' not found in annotation index {i}: {annotation}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_info = self.annotations[idx]
        img_path = os.path.join(self.image_folder, img_info['image'])
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for anno in img_info['annotations']:
            x_center = anno['coordinates']['x']
            y_center = anno['coordinates']['y']
            width = anno['coordinates']['width']
            height = anno['coordinates']['height']
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Assuming 'License_Plate' is the only label with ID 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            image = self.transforms(image)

        return image, target

# 輸入影像
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 定義collate函數(用於整理數據)
def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    # 指定出數據以及註釋的路徑
    image_folder = r'C:\Users\User\Desktop\carcountlicense\License Plate Recognition.v4-resized640_aug3x-accurate.createml\train'
    annotation_file = r'C:\Users\User\Desktop\carcountlicense\License Plate Recognition.v4-resized640_aug3x-accurate.createml\train\_annotations.createml.json'
    
    # 檢查註釋及印出做確認
    with open(annotation_file) as f:
        annotations = json.load(f)
        print("Annotations:", annotations[:5])  # print前五個做確認

    dataset = CustomDataset(image_folder=image_folder, annotation_file=annotation_file, transforms=transforms)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # 建立模型
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

    num_classes = 2  
    faster_rcnn_model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    
    in_features = faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #設備
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    faster_rcnn_model.to(device)

    # 學習
    params = [p for p in faster_rcnn_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if print_freq > 0:
                print(f"Loss: {losses.item()}")

    # 訓練迴圈
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(faster_rcnn_model, optimizer, data_loader, device, epoch)
        lr_scheduler.step()
        print(f"Epoch {epoch + 1} completed.")