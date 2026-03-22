# AI Image Authenticity Analyzer v4.0

## 🚀 SETUP (3 steps only!)

### Step 1: Install Backend Packages
```cmd
cd backend
py -3.11 -m pip install fastapi uvicorn python-multipart Pillow numpy PyJWT
```

### Step 2: Start Backend
```cmd
py -3.11 -m uvicorn main:app --reload --port 8000
```
✅ You should see: "Uvicorn running on http://127.0.0.1:8000"

### Step 3: Open Frontend
- Open the `frontend` folder
- Double-click `index.html`
- OR drag `index.html` into Chrome/Edge

That's it! No npm, no React build needed! 🎉

---

## 🔐 Login
- Username: admin
- Password: forensics2025

---

## 🎯 Features
- ✅ SynthID watermark detection (LSB + frequency analysis)
- ✅ EXIF metadata forensics
- ✅ ELA (Error Level Analysis)
- ✅ AI probability score (0-100%)
- ✅ Heatmap visualization
- ✅ Pixel anomaly map
- ✅ Image reconstruction
- ✅ Scan history
- ✅ JWT authentication
- ✅ Export forensic report
- ✅ Interactive Chart.js dashboard
- ✅ NO OpenCV dependency

---

## 📊 How Scoring Works
- 0-30%  → Likely Real
- 30-70% → Suspicious  
- 70%+   → AI Generated

Combined score = AI Detection (50%) + Metadata Risk (25%) + SynthID (25%)
