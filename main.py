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

# --- 自定义 LaMa 类 ---
class CustomLama:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # 这里的路径必须和 Dockerfile 里 COPY 的位置一致
        model_path = "/root/.cache/simple_lama_inpainting/big-lama.pt"
        
        print(f"📦 加载模型文件: {model_path}")
        try:
            # map_location='cpu' 强制在 CPU 上加载权重
            self.model = torch.jit.load(model_path, map_location='cpu')
            self.model.eval()
            self.model.to(self.device)
            print("✅ 模型加载完成 (CPU模式)")
        except Exception as e:
            print(f"❌ 模型文件加载失败: {e}")
            raise e

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        # 1. 记录原始尺寸
        W, H = image.size

        # 2. 计算需要调整的目标尺寸 (必须是 8 的倍数)
        # LaMa 模型要求输入宽和高都必须能被 8 整除
        new_W = (W // 8) * 8
        new_H = (H // 8) * 8
        
        # 如果尺寸不符合要求，或者图片太小，进行调整
        if new_W != W or new_H != H:
            # 使用双线性插值缩放原图
            image = image.resize((new_W, new_H), Image.BILINEAR)
            # mask 必须用最近邻插值，保证边缘清晰
            mask = mask.resize((new_W, new_H), Image.NEAREST)

        # 3. 转换为 Tensor
        image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0
        
        # (H, W, C) -> (C, H, W)
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 阈值处理 mask
        mask_t = (mask_t > 0).float()

        # 4. 推理
        with torch.no_grad():
            output = self.model(image_t, mask_t)

        # 5. 后处理
        output_np = output[0].permute(1, 2, 0).detach().cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        
        result_img = Image.fromarray(output_np)

        # 6. 如果之前调整过尺寸，现在把结果缩放回原始大小
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
    
    # 初始化模型
    lama_model = CustomLama(device='cpu')
    
except Exception as e:
    print(f"❌ 系统初始化失败: {e}")
    import traceback
    traceback.print_exc()

@app.post("/api/remove-watermark")
async def remove_watermark(image: UploadFile = File(...), mask: UploadFile = File(...)):
    if lama_model is None:
        return StreamingResponse(
            io.BytesIO(b"Model Load Failed"), status_code=500, media_type="text/plain"
        )

    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

        # 确保 mask 和 原图尺寸一致（在进入模型前先对齐）
        if original_img.size != mask_img.size:
            mask_img = mask_img.resize(original_img.size)

        # 调用模型 (内部会自动处理 8 的倍数)
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