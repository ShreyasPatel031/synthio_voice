#!/usr/bin/env bash
# Build + deploy stock Kokoro-82M + scored Misaki lexicon to Cloud Run (NVIDIA L4).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-east4}"
BUILD_REGION="${BUILD_REGION:-us-central1}"
REPO="${AR_REPO:-synthio-voice}"
IMAGE_NAME="${IMAGE_NAME:-kokoro-misaki}"
SERVICE="${SERVICE:-kokoro-misaki}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${BUILD_REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}:${TAG}"
LOG_BUCKET="${LOG_BUCKET:-gs://synthio-voice-cloudbuild-scs}"

if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "Set GCP_PROJECT or gcloud config project" >&2
  exit 1
fi

echo "project=${PROJECT} region=${REGION} image=${IMAGE}"

gcloud artifacts repositories describe "${REPO}" --location="${BUILD_REGION}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${REPO}" \
       --repository-format=docker \
       --location="${BUILD_REGION}" \
       --description="Synthio voice container images"

gcloud storage buckets describe "${LOG_BUCKET}" >/dev/null 2>&1 \
  || gcloud storage buckets create "${LOG_BUCKET}" \
       --location="${BUILD_REGION}" --uniform-bucket-level-access

gcloud builds submit "${ROOT}/services/kokoro_misaki" \
  --tag "${IMAGE}" \
  --timeout=3600s \
  --machine-type=e2-highcpu-8 \
  --gcs-log-dir="${LOG_BUCKET}/logs" \
  --gcs-source-staging-dir="${LOG_BUCKET}/source"

gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --cpu=4 \
  --memory=16Gi \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --no-cpu-throttling \
  --concurrency=1 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=2 \
  --set-env-vars="KOKORO_VOICE=af_heart,KOKORO_SPEED=1.0,KOKORO_SPEED_LOCK=1" \
  --quiet

URL="$(gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')"
echo "SERVICE_URL=${URL}"
echo "Smoke: curl -sS ${URL}/healthz"
