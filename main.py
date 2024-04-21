import torch
from diffusers import StableDiffusionPipeline

random_seed = torch.manual_seed(42)
number_inference_steps = 25
guidance_sacle = 0.75
height = 512
width = 512

#List model
model_list = ["nota-ai/bk-sdm-small",
              "hakurei/waifu-diffusion",
              "stabilityai/stable-diffusion-2-1"
              ]
def create_pipeline(modelname = model_list[1]):
    if torch.cuda.is_available():
        print("Using GPU !!!")
        pipeline = StableDiffusionPipeline.from_pretrained(
            modelname,
            torch_dtype = torch.float16,
            use_safetensors = True
        )
    else:
        print("Using CPU!")
        pipeline = StableDiffusionPipeline.from_pretrained(
            modelname,
            torch_dtype = torch.float32,
            use_safetensors = True
        )
    return pipeline
def text2img(prompt,pipeline):
    img = pipeline(
        prompt,
        guidance_sacle = guidance_sacle,
        number_inference_steps = number_inference_steps,
        generator  = random_seed,
        num_images_per_request = 1,
        width = width,
        height = height
    ).images
    return img[0]

