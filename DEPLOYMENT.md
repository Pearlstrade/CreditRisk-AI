# AI-04 Deployment Guide

## Option A — Streamlit Community Cloud

This is the simplest route for the MVP.

### Git

```bash
git init
git add .
git commit -m "Build AI-04 loan default risk predictor"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai04-loan-default-risk.git
git push -u origin main
```

Then create the app in Streamlit Community Cloud using:

- Repository: `YOUR-USERNAME/ai04-loan-default-risk`
- Branch: `main`
- Main file: `app.py`

No secret/API key is required for this educational MVP.

## Option B — Render

The repository already includes `render.yaml` and a production-style Dockerfile.

1. Push the repository to GitHub.
2. Connect the repository to Render.
3. Render detects `render.yaml`.
4. Deploy the Docker web service.
5. Wait for the health check at `/_stcore/health`.
6. Open the generated `.onrender.com` URL.

## Option C — Any Docker host

```bash
docker build -t ai04-loan-risk .
docker run --restart unless-stopped -p 8501:8501 ai04-loan-risk
```

For a VPS, put Nginx/Caddy in front of Streamlit and enable HTTPS.

## Last-mile production controls

Before accepting real borrower data:

- Add authentication.
- Do not store names, phone numbers, BVN, NIN or other direct identifiers in model inputs unless genuinely required and lawfully processed.
- Add encryption at rest and in transit.
- Add audit logging.
- Add data retention/deletion rules.
- Test for subgroup performance and disparate error rates.
- Recalibrate using representative Nigerian lending outcomes.
- Establish a model-change approval process.
- Keep human review for lending decisions.
