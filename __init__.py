from .qwen2_audio_nodes import Qwen2_AQA

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen2_AQA": Qwen2_AQA,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2_AQA": "Qwen2 AQA",
}