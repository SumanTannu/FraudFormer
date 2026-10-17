# FraudFormer React UI

Build the React application once:

```powershell
cd web
npm install
npm run build
cd ..
```

Then start the complete application from the repository root:

```powershell
python src/prediction_api.py
```

Open `http://127.0.0.1:8000`. The first prediction for a newly selected model can
take longer because its checkpoint is loaded lazily. Only one model is held in
memory at a time.

For frontend development, run `npm run dev` inside `web/`; Vite proxies `/api` to
the Python service on port 8000.
