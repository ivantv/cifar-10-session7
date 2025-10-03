import torch
import torch.nn as nn
import torch.nn.functional as F


dropout_value = 0.05


class ConvBlock(nn.Module):
	def __init__(self, in_c, out_c, k=3, s=1, p=1, use_bn=True):
		super().__init__()
		layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=False)]
		if use_bn:
			layers.append(nn.BatchNorm2d(out_c))
		layers.append(nn.ReLU(inplace=True))
		self.block = nn.Sequential(*layers)

	def forward(self, x):
		return self.block(x)


class Net(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()
		# C1
		self.c1 = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout(dropout_value),
			nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout(dropout_value),
		)
		# C2
		self.c2 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout(dropout_value),
			nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout(dropout_value),
		)
		# C3
		self.c3 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout(dropout_value),
			nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout(dropout_value),
		)
		# C4 (downsample via stride=2 on second conv), then larger kernels to boost RF
		self.c4 = nn.Sequential(
			ConvBlock(128, 256, 3, 1, 1),
			ConvBlock(256, 256, 3, 2, 1),
			ConvBlock(256, 256, 7, 1, 3),
			ConvBlock(256, 256, 5, 1, 2),
			ConvBlock(256, 256, 5, 1, 2)
		)
		self.gap = nn.AdaptiveAvgPool2d((1, 1))
		self.classifier = nn.Conv2d(256, num_classes, 1, 1, 0, bias=False)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		x = self.gap(x)
		x = self.classifier(x)
		x = x.view(x.size(0), -1)
		return F.log_softmax(x, dim=1)


__all__ = ["Net", "ConvBlock"]
