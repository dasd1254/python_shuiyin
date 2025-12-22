import os
# 1. 强制禁用 CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
import torch
import numpy as np

# 设置 CPU 线程数，避免过多线程导致上下文切换变慢
# 根据你的服务器核心数调整，通常 4 或 8 比较合适
torch.set_num_threads(4)

class CustomLama:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        model_path = "/root/.cache/simple_lama_inpainting/big-lama.pt"
        
        print(f"📦 加载模型文件: {model_path}")
        try:
            self.model = torch.jit.load(model_path, map_location='cpu')
            self.model.eval()
            self.model.to(self.device)
            print("✅ 模型加载完成 (CPU模式)")
        except Exception as e:
            print(f"❌ 模型文件加载失败: {e}")
            raise e

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        W, H = image.size
        
        # --- 🚀 性能优化核心逻辑 Start ---
        # 限制最大边长。如果图片太大，强制缩小到 720px 处理
        # 这在 CPU 上能带来 5-10 倍的速度提升，且肉眼几乎看不出质量损失
        MAX_SIZE = 720 
        
        scale_factor = 1.0
        if max(W, H) > MAX_SIZE:
            scale_factor = MAX_SIZE / max(W, H)
            # 临时缩小的尺寸
            temp_W = int(W * scale_factor)
            temp_H = int(H * scale_factor)
        else:
            temp_W, temp_H = W, H

        # 确保尺寸是 8 的倍数 (LaMa 要求)
        process_W = (temp_W // 8) * 8
        process_H = (temp_H // 8) * 8
        
        # 缩放图片和 Mask 用于推理
        img_resized = image.resize((process_W, process_H), Image.BILINEAR)
        mask_resized = mask.resize((process_W, process_H), Image.NEAREST) # Mask 必须用最近邻插值
        # --- 🚀 性能优化核心逻辑 End ---

        # 转换为 Tensor
        image_np = np.array(img_resized.convert("RGB")).astype(np.float32) / 255.0
        mask_np = np.array(mask_resized.convert("L")).astype(np.float32) / 255.0
        
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_t = (mask_t > 0).float()

        # 推理
        with torch.no_grad():
            output = self.model(image_t, mask_t)

        # 后处理
        output_np = output[0].permute(1, 2, 0).detach().cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        
        result_img = Image.fromarray(output_np)

        # 恢复原始尺寸 (如果不恢复，用户下载的就是小图)
        if result_img.size != (W, H):
            result_img = result_img.resize((W, H), Image.BILINEAR)
            
        return result_img

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
    lama_model = CustomLama(device='cpu')
except Exception as e:
    print(f"❌ 系统初始化失败: {e}")

@app.post("/api/remove-watermark")
async def remove_watermark(image: UploadFile = File(...), mask: UploadFile = File(...)):
    if lama_model is None:
        return StreamingResponse(io.BytesIO(b"Model Load Failed"), status_code=500)

    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

        # 对齐尺寸
        if original_img.size != mask_img.size:
            mask_img = mask_img.resize(original_img.size)

        result_img = lama_model(original_img, mask_img)

        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(f"推理错误: {e}")
        return StreamingResponse(io.BytesIO(f"Error: {str(e)}".encode()), status_code=500)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3001)