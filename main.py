from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

# 1. 初始化 APP
app = FastAPI()

# 2. 配置跨域 (允许前端访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 准备目录
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 4. 挂载静态目录 (让前端能通过 URL 访问图片)
app.mount("/result", StaticFiles(directory=PROCESSED_DIR), name="result")

# 5. 初始化 AI 模型 (只加载一次，速度快)
print("正在加载 AI 模型...", flush=True)
ocr = PaddleOCR(
    use_angle_cls=False, 
    lang="ch", 
    ocr_version='PP-OCRv4',
    use_gpu=False,
    enable_mkldnn=False,
    use_mp=False # 关闭多进程省内存
)
print("模型加载完毕！", flush=True)

@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    try:
        # A. 保存上传的文件
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # B. 读取图片
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="无法读取图片文件")

        # C. AI 识别 (直接调用，不需要命令行)
        result = ocr.ocr(file_path, cls=False)

        # D. 去水印逻辑 (如果没有文字，直接返回原图)
        if result is None or len(result) == 0 or result[0] is None:
            output_filename = f"processed_{file.filename}"
            output_path = os.path.join(PROCESSED_DIR, output_filename)
            cv2.imwrite(output_path, img)
            return {
                "code": 200, 
                "msg": "未检测到水印，返回原图", 
                "data": {"url": f"/result/{output_filename}"}
            }

        # E. 创建掩膜并修复
        mask = np.zeros(img.shape[:2], np.uint8)
        for line in result[0]:
            points = np.array(line[0]).astype(np.int32)
            x, y, w, h = cv2.boundingRect(points)
            pad = 5
            cv2.rectangle(mask, (max(0, x-pad), max(0, y-pad)), 
                          (min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)), 255, -1)

        result_img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

        # F. 保存结果
        output_filename = f"processed_{file.filename}"
        output_path = os.path.join(PROCESSED_DIR, output_filename)
        cv2.imwrite(output_path, result_img)

        # G. 返回给前端
        return {
            "code": 200, 
            "msg": "处理成功", 
            "data": {"url": f"/result/{output_filename}"}
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"内部错误: {str(e)}"})

# 本地测试启动命令: python main.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001)