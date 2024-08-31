import os
import folder_paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = folder_paths.get_input_directory()
output_dir = folder_paths.get_output_directory()

class AudioLoader:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1] in ["wav", "mp3", "ogg", "flac", "aiff", "aif"]]
        return {"required":{
            "audio":(files,),
        }}
    
    CATEGORY = "ComfyUI_Qwen2-Audio-7B-Instruct-Int4"
    DESCRIPTION = "Load Audio"

    RETURN_TYPES = ("PATH",)

    OUTPUT_NODE = False

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = os.path.join(input_dir,audio)
        return (audio_path,)

class AudioPreviewer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "audio":("PATH",),
        }}
    
    CATEGORY = "ComfyUI_Qwen2-Audio-7B-Instruct-Int4"
    DESCRIPTION = "Load Audio"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_name = os.path.basename(audio)
        audio_path_name = os.path.basename(os.path.dirname(audio))
        return {"ui":{"audio":[audio_name, audio_path_name]}}
