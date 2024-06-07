from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, AutoencoderKL
import torch
from torch import autocast
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--num", "-n", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()
# np.random.seed(args.seed)

os.makedirs("output_images/", exist_ok=True)

model_id = ""
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
g_cuda = None
pipe.load_lora_weights(model_id)

# prompt: 出力したい画像のプロンプトを書いてください（詳しいことはGoogleで検索すること）
prompt = ""

"""
negative prompot: 質の高い偽画像を生成するためのプロンプト
ただし、人間が見ると画像の質が上がったように感じるが、機械学習の場合はそうでない可能性があることを留意すること
デフォルトでは、negative promptはオフになっているので、negative promptを試したい場合はpipe().imagesにあるコメントアウトを解除する
"""
negative_prompt = "low quality, worst quality, out of focus, ugly, error, lowers, blurry, bokeh" #@param {type:"string"}

# other setting
# 出力する画像のpromptに対する忠実度を指定する（デフォルではオフ、変更する場合はpipe().imagesにあるコメントアウトを解除する。詳細についてはGoogleで検索すること）
guidance_scale = 7.5
# 画像を出力する際に設定する学習回数（多すぎると過学習するので注意が必要）
num_inference_steps = 50 
# 出力する画像の解像度を設定する（変更する場合はpipe().imagesにあるコメントアウトを解除する）
height = 512    # 出力する画像の高さ（デフォルトでは学習時と同じ高さになる）
width = 512     # 出力する画像の幅（デフォルでは学習時と同じ幅になる）

def null_safety(images, **kwargs):
    return images, None
pipe.safety_checker = null_safety

# select random seed
seeds = np.random.randint(0, 65536, args.num)

# seedを毎回変更して生成する方法
for j in range(args.num):
    # seed setting
    g_cuda = torch.Generator(device='cuda')
    seed = seeds[j] #@param {type:"number"}
    seed = seed.item()
    print("seed: "+ str(seed))
    g_cuda.manual_seed(seed)    # seedを設定する
    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt, 
            # height=height, 
            # width=width, 
            # negative_prompt=negative_prompt, 
            num_inference_steps=num_inference_steps, 
            # guidance_scale=guidance_scale, 
            generator=g_cuda).images

    for i, img in enumerate(images):
        img.save('output_images/'+str(j).zfill(4)+'.png')


# seedを固定して生成する方法
# g_cuda = torch.Generator(device='cuda')
# g_cuda.manual_seed(4245)
# images = [
#     pipe(prompt, num_inference_steps=num_inference_steps, generator=g_cuda).images[0]
#     for _ in range(args.num)
# ]

# for i, img in enumerate(images):
#     img.save('output_images/'+str(i).zfill(4)+'.png')