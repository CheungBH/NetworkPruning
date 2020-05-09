from efficientnet_pytorch import EfficientNet
from src.compute_flops import print_model_param_flops

model = EfficientNet.from_name("efficientnet-b0")
print(model)
num = print_model_param_flops(model)
print(num)
