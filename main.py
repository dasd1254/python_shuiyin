from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录准备
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)
app.mount("/result", StaticFiles(directory="processed"), name="result")

# 初始化模型 (轻量版)
print("正在加载 AI 模型...", flush=True)
ocr = PaddleOCR(use_angle_cls=False, lang="ch", ocr_version='PP-OCRv4', use_gpu=False, enable_mkldnn=False)

@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    try:
        # 保存文件
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 读取图片
        img = cv2.imread(file_path)
        
        # 识别
        result = ocr.ocr(file_path, cls=False)
        
        # 如果没水印，直接返回
        if not result or not result[0]:
            shutil.copy(file_path, f"processed/processed_{file.filename}")
            return {"code": 200, "data": {"url": f"/result/processed_{file.filename}"}}

        # 去水印
        mask = np.zeros(img.shape[:2], np.uint8)
        for line in result[0]:
            points = np.array(line[0]).astype(np.int32)
            x, y, w, h = cv2.boundingRect(points)
            pad = 5
            cv2.rectangle(mask, (max(0, x-pad), max(0, y-pad)), (min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)), 255, -1)
        
        res_img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        
        # 保存结果
        out_name = f"processed_{file.filename}"
        cv2.imwrite(f"processed/{out_name}", res_img)
        
        return {"code": 200, "msg": "成功", "data": {"url": f"/result/{out_name}"}}
    except Exception as e:
        print(f"Error: {e}")
        return {"code": 500, "msg": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001)