from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # <--- 关键！之前漏了这行
from simple_lama_inpainting import SimpleLama
from PIL import Image
import io
import uvicorn

app = FastAPI()

# ================= 新增跨域配置 =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有 Header
)
# ===============================================

# 1. 初始化模型
print("正在加载 LaMa AI 模型，请稍候...")
try:
    lama_model = SimpleLama()
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")

@app.post("/api/remove-watermark")
async def remove_watermark(
    image: UploadFile = File(...), 
    mask: UploadFile = File(...)
):
    try:
        # 1. 读取原图
        image_bytes = await image.read()
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. 读取蒙版图
        mask_bytes = await mask.read()
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L") 

        # 3. 验证尺寸
        if original_img.size != mask_img.size:
            mask_img = mask_img.resize(original_img.size)

        # 4. 执行 AI 修复
        result_img = lama_model(original_img, mask_img)

        # 5. 返回图片
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(f"Error: {str(e)}")
        # 注意：StreamingResponse 出错时最好返回 JSON 提示，但前端要做好兼容
        # 这里简单起见，为了防止前端解析二进制失败，可以在控制台打印错误
        return {"code": 500, "msg": f"处理失败: {str(e)}"}

if __name__ == '__main__':
    # 必须是 0.0.0.0
    uvicorn.run(app, host='0.0.0.0', port=3001)