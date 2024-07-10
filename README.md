# Spatio-Temporal Shuffle Guidance

## Comparison

| Unconditional  | SVD            | STSG           |
| -------------- | -------------- | -------------- |
| <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/11f7928f-0283-43f9-8f4d-fc2937c1707d" width="400"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/fd613ae3-c16c-42e0-90b2-29f150a39895" width="400"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/69286a0a-49f0-4a09-a474-17b14d777d3d" width="400"> |
| <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/3893d187-623a-4801-a8fa-4ce6126d41e5" width="400"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/95d48275-9403-47d7-a43d-22e5fc8f9877" width="400"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/41efbf33-7ab7-4784-aa98-1a54fbf64806" width="400"> |

## Using STSG with Stable Video Diffusion

```python
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
```