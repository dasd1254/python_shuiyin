from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from simple_lama_inpainting import SimpleLama
from PIL import Image
import io
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lama_model = None
MODEL_PATH = "/root/.cache/torch/hub/checkpoints/big-lama.pt"

print("------------- 系统启动检查 -------------")
# 1. 检查模型文件是否存在及大小
if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ 发现模型文件: {MODEL_PATH}")
    print(f"📄 文件大小: {size_mb:.2f} MB (正常应约为 196 MB)")
    if size_mb < 100:
        print("⚠️ 警告：模型文件过小，可能下载不完整！")
else:
    print(f"❌ 未找到模型文件: {MODEL_PATH}")
    # 打印一下当前目录看看文件在哪
    print(f"当前目录文件: {os.listdir('.')}")

try:
    print("🚀 正在加载 LaMa 模型...")
    # 强制不使用环境变量，依赖默认路径
    if 'TORCH_HOME' in os.environ:
        del os.environ['TORCH_HOME']
    
    lama_model = SimpleLama()
    print("✅ 模型加载成功！服务已就绪。")
except Exception as e:
    print(f"❌ 模型加载崩溃: {e}")

@app.post("/api/remove-watermark")
async def remove_watermark(image: UploadFile = File(...), mask: UploadFile = File(...)):
    if lama_model is None:
        # 返回 500 状态码，这样前端能捕获到错误
        return StreamingResponse(
            io.BytesIO(b"Model not loaded"), 
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

        result_img = lama_model(original_img, mask_img)

        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(f"处理出错: {e}")
        return StreamingResponse(
            io.BytesIO(str(e).encode()), 
            status_code=500, 
            media_type="text/plain"
        )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3001)