from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from simple_lama_inpainting import SimpleLama
from PIL import Image
import io
import uvicorn
import os

app = FastAPI()

# ================= 跨域配置 =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 模型加载逻辑 (关键修改) =================
lama_model = None  # 先定义全局变量，防止 NameError
MODEL_LOAD_ERROR = None # 记录具体的加载错误信息

print("正在初始化 LaMa 模型...")
try:
    # 尝试加载模型
    lama_model = SimpleLama()
    print("✅ 模型加载成功！")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print(f"❌ 模型加载失败: {e}")
    # 打印一下环境变量和当前目录，方便排查路径问题
    print(f"TORCH_HOME: {os.environ.get('TORCH_HOME')}")
    print(f"当前目录文件: {os.listdir('.')}")

@app.post("/api/remove-watermark")
async def remove_watermark(
    image: UploadFile = File(...), 
    mask: UploadFile = File(...)
):
    # 1. 检查模型是否就绪
    if lama_model is None:
        return {
            "code": 500, 
            "msg": f"服务启动失败，模型未能加载。错误详情: {MODEL_LOAD_ERROR}"
        }

    try:
        # 2. 读取原图
        image_bytes = await image.read()
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 3. 读取蒙版图
        mask_bytes = await mask.read()
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L") 

        # 4. 验证尺寸
        if original_img.size != mask_img.size:
            print(f"调整尺寸: 原图{original_img.size} vs 蒙版{mask_img.size}")
            mask_img = mask_img.resize(original_img.size)

        # 5. 执行 AI 修复
        print("开始推理...")
        result_img = lama_model(original_img, mask_img)
        print("推理完成")

        # 6. 返回图片
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(f"处理异常: {str(e)}")
        return {"code": 500, "msg": f"图片处理失败: {str(e)}"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3001)