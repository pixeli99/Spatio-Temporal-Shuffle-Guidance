# Spatio-Temporal Shuffle Guidance

## Comparison

### Conditional Generation
|Init Frame| SVD(w/o CFG)  | SVD            | STSG           |
| -------------- | -------------- | -------------- | -------------- |
| <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/5a32105e-b17f-4695-80c7-2a0562f0a6da" width="300" height="150"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/11f7928f-0283-43f9-8f4d-fc2937c1707d" width="300" height="150"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/fd613ae3-c16c-42e0-90b2-29f150a39895" width="300" height="150"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/69286a0a-49f0-4a09-a474-17b14d777d3d" width="300" height="150"> |
| <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/308228f2-7cc0-43cb-95ef-2fbe14ae9d05" width="300" height="150"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/3893d187-623a-4801-a8fa-4ce6126d41e5" width="300" height="150"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/95d48275-9403-47d7-a43d-22e5fc8f9877" width="300" height="150"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/41efbf33-7ab7-4784-aa98-1a54fbf64806" width="300" height="150"> |

### Unconditional Generation
Below is the result generated without the `conditioning image`.
| SVD            | STSG           |
| -------------- | -------------- |
| <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/72b7e123-99c6-4ab4-a364-787120fd164a" width="300" height="150"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/85d630ad-9fbf-45f5-87e8-6534b4590f8c" width="300" height="150"> |
| <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/675de320-97cc-4c23-9f32-b7814096ea55" width="300" height="150"> | <img src="https://github.com/pixeli99/Spatio-Temporal-Shuffle-Guidance/assets/46072190/9fcf9831-dfe1-4708-8e06-0a51b428799e" width="300" height="150"> |
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