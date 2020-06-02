import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from base.base_model import BaseModel
from .backbones import resnet
from .backbones import xception
from .backbones import mobilenetv2


class _ASPPModule(nn.Module):
	def __init__(self, inplanes, planes, kernel_size, padding, dilation):
		super(_ASPPModule, self).__init__()
		self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
		self.bn = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.atrous_conv(x)
		x = self.bn(x)
		return self.relu(x)


class ASPP(nn.Module):
	def __init__(self, output_stride, inplanes):
		super(ASPP, self).__init__()

		if output_stride == 16:
			dilations = [1, 6, 12, 18]
		elif output_stride == 8:
			dilations = [1, 12, 24, 36]

		self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
		self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
		self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
		self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

		self.global_avg_pool = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(256)
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		x1 = self.aspp1(x)
		x2 = self.aspp2(x)
		x3 = self.aspp3(x)
		x4 = self.aspp4(x)
		x5 = self.global_avg_pool(x)
		x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x1, x2, x3, x4, x5), dim=1)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		return self.dropout(x)

class Decoder(nn.Module):
	def __init__(self, num_classes, low_level_inplanes):
		super(Decoder, self).__init__()

		self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(48)
		self.relu = nn.ReLU(inplace=True)
		self.last_conv = nn.Sequential(
			nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
		)

	def forward(self, x, low_level_feat):
		low_level_feat = self.conv1(low_level_feat)
		low_level_feat = self.bn1(low_level_feat)
		low_level_feat = self.relu(low_level_feat)

		x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x, low_level_feat), dim=1)
		x = self.last_conv(x)
		return x


class DeepLabV3Plus(BaseModel):
	def __init__(self, backbone='resnet50', output_stride=16, num_classes=2, freeze_bn=False, pretrained_backbone=None):
		super(DeepLabV3Plus, self).__init__()
		if 'resnet' in backbone:
			if backbone=='resnet18':
				num_layers = 18
				inplanes = 512
				low_level_inplanes = 64
			elif backbone=='resnet34':
				num_layers = 34
				inplanes = 512
				low_level_inplanes = 64
			elif backbone=='resnet50':
				num_layers = 50
				inplanes = 2048
				low_level_inplanes = 256
			elif backbone=='resnet101':
				num_layers = 101
				inplanes = 2048
				low_level_inplanes = 256

			self.backbone = resnet.get_resnet(num_layers=num_layers, num_classes=None)
			self._run_backbone = self._run_backbone_resnet
			self.aspp = ASPP(output_stride, inplanes=inplanes)
			self.decoder = Decoder(num_classes, low_level_inplanes=low_level_inplanes)

		elif backbone=='mobilenetv2':
			alpha = 1.0
			expansion = 6
			self.backbone = mobilenetv2.MobileNetV2(alpha=alpha, expansion=expansion, num_classes=None)
			self._run_backbone = self._run_backbone_mobilenetv2
			self.aspp = ASPP(output_stride, inplanes=1280)
			self.decoder = Decoder(num_classes, low_level_inplanes=24)
		
		elif backbone=='xception65':
			self.backbone = xception.xception65()
			self._run_backbone = self._run_backbone_xception65
			self.aspp = ASPP(output_stride, inplanes=2048)
			self.decoder = Decoder(num_classes, low_level_inplanes=256)
		else:
			raise NotImplementedError

		self._init_weights()
		if pretrained_backbone is not None:
			self.backbone.load_pretrained_model(pretrained_backbone)
		if freeze_bn:
			self._freeze_bn()


	def forward(self, input):
		x, low_feat = self._run_backbone(input)
		x = self.aspp(x)
		x = self.decoder(x, low_feat)
		x = F.interpolate(x, size=input.shape[-2:], mode='bilinear', align_corners=True)
		return x


	def _run_backbone_resnet(self, input):
		# Stage1
		x1 = self.backbone.conv1(input)
		x1 = self.backbone.bn1(x1)
		x1 = self.backbone.relu(x1)
		# Stage2
		x2 = self.backbone.maxpool(x1)
		x2 = self.backbone.layer1(x2)
		# Stage3
		x3 = self.backbone.layer2(x2)
		# Stage4
		x4 = self.backbone.layer3(x3)
		# Stage5
		x5 = self.backbone.layer4(x4)
		# Output
		return x5, x2
	
	def _run_backbone_mobilenetv2(self, input):
		x = input
		# Stage1
		x = reduce(lambda x, n: self.backbone.features[n](x), list(range(0,2)), x)
		x1 = x
		# Stage2
		x = reduce(lambda x, n: self.backbone.features[n](x), list(range(2,4)), x)
		x2 = x
		# Stage3
		x = reduce(lambda x, n: self.backbone.features[n](x), list(range(4,7)), x)
		x3 = x
		# Stage4
		x = reduce(lambda x, n: self.backbone.features[n](x), list(range(7,14)), x)
		x4 = x
		# Stage5
		x5 = reduce(lambda x, n: self.backbone.features[n](x), list(range(14,19)), x)
		return x5, x2

	def _run_backbone_xception65(self, input):
		x = input
		# Entry flow
		x = self.backbone.conv1(x)
		x = self.backbone.bn1(x)
		x = self.backbone.relu(x)

		x = self.backbone.conv2(x)
		x = self.backbone.bn2(x)
		x = self.backbone.relu(x)

		x = self.backbone.block1(x)
		x, c1 = self.backbone.block2(x)  # b, h//4, w//4, 256
		x, c2 = self.backbone.block3(x)  # b, h//8, w//8, 728

		# Middle flow
		x = self.backbone.block4(x)
		x = self.backbone.block5(x)
		x = self.backbone.block6(x)
		x = self.backbone.block7(x)
		x = self.backbone.block8(x)
		x = self.backbone.block9(x)
		x = self.backbone.block10(x)
		x = self.backbone.block11(x)
		x = self.backbone.block12(x)
		x = self.backbone.block13(x)
		x = self.backbone.block14(x)
		x = self.backbone.block15(x)
		x = self.backbone.block16(x)
		x = self.backbone.block17(x)
		x = self.backbone.block18(x)
		c3 = self.backbone.block19(x)

		# Exit flow
		x = self.backbone.block20(c3)
		c4 = self.backbone.block21(x)
		return c4,c1

	def _freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()


	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)