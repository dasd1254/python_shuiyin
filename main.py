import os
# 1. 强制禁用 CUDA，防止 PyTorch 抱有幻想
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
import torch
import numpy as np
from typing import Union

# --- 自定义 LaMa 类 (替代 simple_lama_inpainting) ---
class CustomLama:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # 模型下载路径
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "simple_lama_inpainting")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        model_path = os.path.join(cache_dir, "big-lama.pt")
        
        # 自动下载模型 (如果不存在)
        if not os.path.exists(model_path):
            print(f"📥 正在下载模型到 {model_path} ...")
            url = "https://github.com/madebyollin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
            torch.hub.download_url_to_file(url, model_path)
        
        print(f"📦 加载模型文件: {model_path}")
        try:
            # 【关键修复】map_location='cpu' 强制在 CPU 上加载权重
            self.model = torch.jit.load(model_path, map_location='cpu')
            self.model.eval()
            self.model.to(self.device)
            print("✅ 模型加载完成 (CPU模式)")
        except Exception as e:
            print(f"❌ 模型文件加载失败: {e}")
            raise e

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        # 预处理：调整图片尺寸以符合 8 的倍数 (LaMa 的要求)
        W, H = image.size
        # 简单的填充或调整大小逻辑，这里简化处理，直接推理
        
        # 转换为 Tensor
        image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0
        
        # (H, W, C) -> (C, H, W)
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 阈值处理 mask
        mask_t = (mask_t > 0).float()

        # 推理
        with torch.no_grad():
            output = self.model(image_t, mask_t)

        # 后处理
        output_np = output[0].permute(1, 2, 0).detach().cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(output_np)

# --- FastAPI App ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lama_model = None

print("------------- 系统启动 -------------")
try:
    if 'TORCH_HOME' in os.environ:
        del os.environ['TORCH_HOME']
    
    # 使用我们自定义的类，而不是库里的
    lama_model = CustomLama(device='cpu')
    
except Exception as e:
    print(f"❌ 系统初始化失败: {e}")
    # 打印完整报错以便调试
    import traceback
    traceback.print_exc()

@app.post("/api/remove-watermark")
async def remove_watermark(image: UploadFile = File(...), mask: UploadFile = File(...)):
    if lama_model is None:
        return StreamingResponse(
            io.BytesIO(b"Model Load Failed - Check Server Logs"), 
            status_code=500, 
            media_type="text/plain"
        )

    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

        # 确保 mask 和 原图尺寸一致
        if original_img.size != mask_img.size:
            mask_img = mask_img.resize(original_img.size)

        result_img = lama_model(original_img, mask_img)

        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(f"推理错误: {e}")
        import traceback
        traceback.print_exc()
        return StreamingResponse(
            io.BytesIO(f"Runtime Error: {str(e)}".encode()), 
            status_code=500, 
            media_type="text/plain"
        )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3001)