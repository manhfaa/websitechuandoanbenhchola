# LeafCare AI (Split Frontend + Backend)

Du an da duoc tach thanh 2 phan de deploy rieng tren Render, giup giam kich thuoc moi service (phu hop gioi han 512 MB):

- `backend/`: Flask API + model YOLO/CNN + xu ly anh
- `frontend/`: giao dien HTML/CSS/JS goi API backend

## Kien truc

1. Frontend goi `GET /api/health` de kiem tra trang thai.
2. Frontend goi `POST /api/analyze` de upload anh.
3. Backend tra ket qua va URL anh xu ly tai `/uploads/...`.

## Cau truc thu muc

- `backend/app.py`: API Flask (khong render template)
- `backend/services/*`: pipeline YOLO -> CNN -> LLM
- `backend/model_0.h5`: CNN model
- `backend/moduleyolola/best.pt`: YOLO model
- `frontend/index.html`: UI
- `frontend/static/js/config.js`: cau hinh URL backend
- `frontend/static/js/app.js`: logic upload/camera/goi API
- `render.yaml`: blueprint deploy 2 service tren Render

## Chay local

### 1) Backend

```bash
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Backend mac dinh: `http://localhost:5000`

### 2) Frontend

Sua file `frontend/static/js/config.js`:

```js
window.APP_CONFIG = {
  API_BASE_URL: "http://localhost:5000",
};
```

Mo file `frontend/index.html` bang live server hoac bat ky static server nao.

## Deploy Render tu A-Z

1. Push toan bo repo len GitHub.
2. Tren Render, chon `New +` -> `Blueprint`.
3. Chon repo nay, Render doc `render.yaml` va tao 2 service:
   - `leafcare-backend` (Python web service)
   - `leafcare-frontend` (Static site)
4. Nhap secret `OPENAI_API_KEY` cho backend khi duoc hoi.
5. Sau khi deploy lan dau, mo frontend URL va thu tai anh.
6. Kiem tra backend health: `https://leafcare-backend.onrender.com/api/health`

## Bien moi truong quan trong

- Backend:
  - `OPENAI_API_KEY` (secret)
  - `FRONTEND_ORIGIN` (mac dinh trong `render.yaml`: `https://leafcare-frontend.onrender.com`)
- Frontend:
  - Cap nhat `frontend/static/js/config.js` neu backend URL thay doi.

## Ghi chu dung luong

- Frontend duoc deploy rieng, khong kem model ML.
- Backend chi build tu `backend/` nho `rootDir`, tranh mang theo tai nguyen khong can thiet cua frontend.
- File tam trong `backend/uploads/*` da duoc bo khoi repo (chi giu `.gitkeep`).
