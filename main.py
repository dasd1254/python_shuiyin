import os
# 1. 强制禁用 CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # 新增：用于定义 JSON 数据结构
from PIL import Image
import io
import uvicorn
import torch
import numpy as np
import base64 # 新增：用于解码

# 设置 CPU 线程数
torch.set_num_threads(4)

class CustomLama:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        # 请确认这个路径是正确的，且文件存在
        model_path = "/root/.cache/simple_lama_inpainting/big-lama.pt"
        
        print(f"📦 加载模型文件: {model_path}")
        try:
            self.model = torch.jit.load(model_path, map_location='cpu')
            self.model.eval()
            self.model.to(self.device)
            print("✅ 模型加载完成 (CPU模式)")
        except Exception as e:
            print(f"❌ 模型文件加载失败: {e}")
            # 这里如果不抛出异常，后面调用会报错，建议抛出或处理
            # raise e 

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        W, H = image.size
        
        # --- 🚀 性能优化核心逻辑 Start ---
        MAX_SIZE = 720 
        scale_factor = 1.0
        if max(W, H) > MAX_SIZE:
            scale_factor = MAX_SIZE / max(W, H)
            temp_W = int(W * scale_factor)
            temp_H = int(H * scale_factor)
        else:
            temp_W, temp_H = W, H

        process_W = (temp_W // 8) * 8
        process_H = (temp_H // 8) * 8
        
        img_resized = image.resize((process_W, process_H), Image.BILINEAR)
        mask_resized = mask.resize((process_W, process_H), Image.NEAREST)
        # --- 🚀 性能优化核心逻辑 End ---

        image_np = np.array(img_resized.convert("RGB")).astype(np.float32) / 255.0
        mask_np = np.array(mask_resized.convert("L")).astype(np.float32) / 255.0
        
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_t = (mask_t > 0).float()

        with torch.no_grad():
            output = self.model(image_t, mask_t)

        output_np = output[0].permute(1, 2, 0).detach().cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        
        result_img = Image.fromarray(output_np)

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

# 定义请求体结构：接收两个 Base64 字符串
class RemoveWatermarkRequest(BaseModel):
    image: str
    mask: str

lama_model = None

print("------------- 系统启动 -------------")
try:
    if 'TORCH_HOME' in os.environ:
        del os.environ['TORCH_HOME']
    lama_model = CustomLama(device='cpu')
except Exception as e:
    print(f"❌ 系统初始化失败: {e}")

# 辅助函数：解码 Base64
def decode_base64(data_str):
    # 如果前端传来的字符串包含 "data:image/png;base64," 前缀，需要去掉
    if ',' in data_str:
        data_str = data_str.split(',')[1]
    return base64.b64decode(data_str)

@app.post("/api/remove-watermark-base64")
async def remove_watermark(data: RemoveWatermarkRequest):
    """
    修改后的接口：接收 JSON Body
    {
        "image": "base64字符串...",
        "mask": "base64字符串..."
    }
    """
    if lama_model is None:
        return StreamingResponse(io.BytesIO(b"Model Load Failed"), status_code=500)

    try:
        # 1. Base64 解码 -> Bytes
        image_bytes = decode_base64(data.image)
        mask_bytes = decode_base64(data.mask)
        
        # 2. Bytes -> PIL Image (后续逻辑完全不变)
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

        # 对齐尺寸
        if original_img.size != mask_img.size:
            mask_img = mask_img.resize(original_img.size)

        # 推理
        result_img = lama_model(original_img, mask_img)

        # 保存结果到内存
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # 返回图片流
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        import traceback
        traceback.print_exc() # 打印详细报错方便调试
        print(f"推理错误: {e}")
        return StreamingResponse(io.BytesIO(f"Error: {str(e)}".encode()), status_code=500)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3001)