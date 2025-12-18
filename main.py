import os
# 1. 【修改点】最关键的一步：在导入 torch 之前，强制禁用 CUDA 显卡可见性
# 这会让 PyTorch 彻底死心，乖乖使用 CPU，不会报 CUDA backend 错误
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from simple_lama_inpainting import SimpleLama
from PIL import Image
import io
import uvicorn
import torch # 建议显式导入 torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lama_model = None

# 打印启动日志
print("------------- 系统启动 -------------")
try:
    # 强制移除环境变量，确保使用默认路径 /root/.cache/torch/...
    if 'TORCH_HOME' in os.environ:
        del os.environ['TORCH_HOME']
    
    print("🚀 开始加载 LaMa 模型 (CPU模式)...")
    
    # 2. 【修改点】显式指定 device='cpu'
    # 即使上面设置了环境变量，这里显式指定会更保险
    lama_model = SimpleLama(device='cpu')
    
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载严重失败: {e}")

@app.post("/api/remove-watermark")
async def remove_watermark(image: UploadFile = File(...), mask: UploadFile = File(...)):
    # 如果模型没加载成功，直接返回 500 状态码
    if lama_model is None:
        return StreamingResponse(
            io.BytesIO(b"Model Load Failed"), 
            status_code=500, 
            media_type="text/plain"
        )

    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

        if original_img.size != mask_img.size:
            mask_img = mask_img.resize(original_img.size)

        # 模型推理
        result_img = lama_model(original_img, mask_img)

        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(f"推理错误: {e}")
        # 打印更详细的错误堆栈，方便排查
        import traceback
        traceback.print_exc()
        
        return StreamingResponse(
            io.BytesIO(f"Error: {e}".encode()), 
            status_code=500, 
            media_type="text/plain"
        )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3001)