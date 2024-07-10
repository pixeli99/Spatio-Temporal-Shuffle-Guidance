import torch
from diffusers.utils import load_image, export_to_video
from pipeline_stable_video_diffusion_stsg import StableVideoDiffusionSTSGipeline

pipe = StableVideoDiffusionSTSGipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("demo.jpg")
image = image.resize((1024, 576))

generator = torch.manual_seed(1)
frames = pipe(image, decode_chunk_size=8, generator=generator, min_guidance_scale=1.0, max_guidance_scale=2.0,).frames[0]
export_to_video(frames, "./res.mp4", fps=7)