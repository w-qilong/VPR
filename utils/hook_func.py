# 使用transformers库中的DINOv2模型
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# load model and processor
checkpoint = "/media/cartolab3/DataDisk/wuqilong_file/Projects/RerenkVPR/pretrained_model/dinov2_small"
processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
print(model)


# 定义一个hook函数，获取多个层的输出
def get_layers_output(layers):
    outputs = {layer: [] for layer in layers}

    def hook(module, input, output, layer_name):
        outputs[layer_name].append(output)

    # 注册hook
    for layer in layers:
        layer.register_forward_hook(lambda module, input, output, layer_name=layer: hook(module, input, output, layer_name))
    
    return outputs

image = Image.open("imgs/0000003.jpg")


# 获取指定层的输出
layers_to_hook = [model.encoder.layer[0].attention.attention.query, 
                  model.encoder.layer[0].attention.attention.key, 
                  model.encoder.layer[0].attention.attention.value]

layers_output = get_layers_output(layers_to_hook)
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
query, key, value = layers_output.values()

print(query[0].shape, key[0].shape, value[0].shape)