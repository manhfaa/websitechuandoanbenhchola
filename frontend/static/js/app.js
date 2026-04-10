const state = {
  file: null,
  previewUrl: "",
  mediaStream: null,
};

const API_BASE_URL = String(window.APP_CONFIG?.API_BASE_URL || window.location.origin).replace(
  /\/$/,
  ""
);

const elements = {
  fileInput: document.getElementById("fileInput"),
  dropzone: document.getElementById("dropzone"),
  previewImage: document.getElementById("previewImage"),
  fileName: document.getElementById("fileName"),
  analyzeButton: document.getElementById("analyzeButton"),
  resetButton: document.getElementById("resetButton"),
  openCameraButton: document.getElementById("openCameraButton"),
  captureButton: document.getElementById("captureButton"),
  closeCameraButton: document.getElementById("closeCameraButton"),
  symptomsInput: document.getElementById("symptomsInput"),
  cameraShell: document.getElementById("cameraShell"),
  cameraVideo: document.getElementById("cameraStream"),
  cameraHelper: document.getElementById("cameraHelper"),
  captureCanvas: document.getElementById("captureCanvas"),
  resultsCard: document.getElementById("resultsCard"),
  messageBanner: document.getElementById("messageBanner"),
  pipelineList: document.getElementById("pipelineList"),
  serverStatus: document.getElementById("serverStatus"),
  healthYolo: document.getElementById("healthYolo"),
  healthCnn: document.getElementById("healthCnn"),
  healthLabels: document.getElementById("healthLabels"),
  healthOpenAi: document.getElementById("healthOpenAi"),
  processingTime: document.getElementById("processingTime"),
  resultOriginal: document.getElementById("resultOriginal"),
  resultAnnotated: document.getElementById("resultAnnotated"),
  resultCrop: document.getElementById("resultCrop"),
  cnnHeadline: document.getElementById("cnnHeadline"),
  cnnLabel: document.getElementById("cnnLabel"),
  cnnConfidence: document.getElementById("cnnConfidence"),
  cnnWarning: document.getElementById("cnnWarning"),
  predictionList: document.getElementById("predictionList"),
  llmSource: document.getElementById("llmSource"),
  llmHeadline: document.getElementById("llmHeadline"),
  llmSummary: document.getElementById("llmSummary"),
  careSteps: document.getElementById("careSteps"),
  nextSteps: document.getElementById("nextSteps"),
  llmWarning: document.getElementById("llmWarning"),
};

const CAMERA_DEFAULT_TEXT =
  "Camera se hoat dong tot tren localhost va ban Render da bat HTTPS.";

function basePipeline() {
  return [
    {
      title: "YOLO nhan dien la",
      detail: "Tach vung la ro nhat truoc khi dua sang CNN.",
    },
    {
      title: "CNN phan loai",
      detail: "Doc anh crop va tinh xac suat cho tung lop cua model_0.h5.",
    },
    {
      title: "ChatGPT tu van",
      detail: "Tom tat ngan gon, de hieu va goi y cham soc tiep theo.",
    },
  ];
}

async function init() {
  renderPipeline(basePipeline());
  bindEvents();
  syncCameraAvailability();
  await loadHealth();
}

function bindEvents() {
  elements.fileInput.addEventListener("change", (event) => {
    const [file] = event.target.files;
    applyFile(file);
  });

  elements.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("drag-over");
  });

  elements.dropzone.addEventListener("dragleave", () => {
    elements.dropzone.classList.remove("drag-over");
  });

  elements.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("drag-over");
    const [file] = event.dataTransfer.files;
    applyFile(file);
  });

  elements.openCameraButton.addEventListener("click", startCamera);
  elements.captureButton.addEventListener("click", captureImage);
  elements.closeCameraButton.addEventListener("click", closeCameraPanel);
  elements.analyzeButton.addEventListener("click", analyzeImage);
  elements.resetButton.addEventListener("click", resetForm);

  window.addEventListener("beforeunload", stopCameraStream);
}

function syncCameraAvailability() {
  if (hasCameraSupport()) {
    return;
  }

  elements.openCameraButton.disabled = true;
  elements.cameraHelper.textContent =
    "Trinh duyet hien tai khong ho tro chup anh truc tiep. Ban van co the tai anh thu cong.";
}

function hasCameraSupport() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function applyFile(file, options = {}) {
  if (!file) {
    return;
  }

  if (file.type && !file.type.startsWith("image/")) {
    showBanner("Vui long chon dung file anh JPG, PNG hoac WEBP.", "error");
    return;
  }

  state.file = file;
  elements.fileName.textContent = options.label || file.name;
  elements.analyzeButton.disabled = false;

  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = state.previewUrl;
  elements.previewImage.classList.remove("is-empty");

  if (options.fromCamera) {
    showBanner("Da chup anh thanh cong. Ban co the bam Phan tich ngay.", "info");
  }
}

function resetForm() {
  state.file = null;

  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.previewUrl = "";
  elements.fileInput.value = "";
  elements.symptomsInput.value = "";
  elements.fileName.textContent = "Chua chon anh";
  elements.previewImage.removeAttribute("src");
  elements.previewImage.classList.add("is-empty");
  elements.analyzeButton.disabled = true;
  clearResults();
  closeCameraPanel();
  hideBanner();
  renderPipeline(basePipeline());
}

function clearResults() {
  elements.resultsCard.classList.add("hidden");
  elements.processingTime.textContent = "0 ms";
  elements.resultOriginal.removeAttribute("src");
  elements.resultAnnotated.removeAttribute("src");
  elements.resultCrop.removeAttribute("src");
  elements.cnnHeadline.textContent = "Chua co du lieu";
  elements.cnnLabel.textContent = "-";
  elements.cnnConfidence.textContent = "Do tin cay: -";
  elements.cnnWarning.textContent = "";
  elements.cnnWarning.classList.add("hidden");
  elements.predictionList.innerHTML = "";
  elements.llmSource.textContent = "Nguon: -";
  elements.llmHeadline.textContent = "-";
  elements.llmSummary.textContent = "-";
  elements.careSteps.innerHTML = "";
  elements.nextSteps.innerHTML = "";
  elements.llmWarning.textContent = "-";
}

async function loadHealth() {
  try {
    const response = await fetch(buildApiUrl("/api/health"));
    const data = await readJsonResponse(response, "Khong doc duoc trang thai backend.");
    const dependencies = data.dependencies;

    elements.serverStatus.textContent = "San sang";
    elements.serverStatus.className = "status-pill success";
    elements.healthYolo.textContent =
      dependencies.yolo_model_found && dependencies.ultralytics_ready
        ? "San sang"
        : dependencies.yolo_model_found
          ? "Thieu ultralytics"
          : "Thieu model";
    elements.healthCnn.textContent =
      dependencies.cnn_model_found && dependencies.tensorflow_ready
        ? "San sang"
        : dependencies.cnn_model_found
          ? "Thieu TensorFlow"
          : "Thieu model";
    elements.healthLabels.textContent = dependencies.cnn_labels_found
      ? "Co file nhan"
      : "Dang dung nhan mau";
    elements.healthOpenAi.textContent = dependencies.openai_key_configured
      ? "Da cau hinh"
      : "Chua co API key";

    if (!dependencies.ultralytics_ready || !dependencies.tensorflow_ready) {
      showBanner(
        "Moi truong hien tai dang thieu mot so thu vien ML. Website van mo duoc, nhung suy luan model co the khong day du.",
        "info"
      );
    }
  } catch (error) {
    elements.serverStatus.textContent = "Khong ket noi";
    elements.serverStatus.className = "status-pill warning";
    elements.healthYolo.textContent = "Khong ro";
    elements.healthCnn.textContent = "Khong ro";
    elements.healthLabels.textContent = "Khong ro";
    elements.healthOpenAi.textContent = "Khong ro";
  }
}

async function startCamera() {
  if (!hasCameraSupport()) {
    showBanner("Trinh duyet khong ho tro camera truc tiep.", "error");
    return;
  }

  hideBanner();
  elements.cameraShell.classList.remove("hidden");
  elements.cameraHelper.textContent = "Dang yeu cau quyen camera...";

  stopCameraStream();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
      },
      audio: false,
    });

    state.mediaStream = stream;
    elements.cameraVideo.srcObject = stream;
    await elements.cameraVideo.play();

    elements.openCameraButton.classList.add("hidden");
    elements.captureButton.classList.remove("hidden");
    elements.closeCameraButton.classList.remove("hidden");
    elements.cameraHelper.textContent = "Camera da san sang. Can la vao giua khung roi bam Chup anh.";
  } catch (error) {
    closeCameraPanel();
    showBanner(
      "Khong mo duoc camera. Hay cap quyen camera cho trinh duyet va thu lai.",
      "error"
    );
  }
}

function stopCameraStream() {
  if (!state.mediaStream) {
    return;
  }

  for (const track of state.mediaStream.getTracks()) {
    track.stop();
  }

  state.mediaStream = null;
  elements.cameraVideo.srcObject = null;
}

function closeCameraPanel() {
  stopCameraStream();
  elements.cameraShell.classList.add("hidden");
  elements.openCameraButton.classList.remove("hidden");
  elements.captureButton.classList.add("hidden");
  elements.closeCameraButton.classList.add("hidden");
  elements.cameraHelper.textContent = CAMERA_DEFAULT_TEXT;
}

async function captureImage() {
  if (!state.mediaStream || !elements.cameraVideo.videoWidth || !elements.cameraVideo.videoHeight) {
    showBanner("Camera chua san sang de chup. Hay doi 1 chut roi thu lai.", "error");
    return;
  }

  const canvas = elements.captureCanvas;
  const context = canvas.getContext("2d");

  canvas.width = elements.cameraVideo.videoWidth;
  canvas.height = elements.cameraVideo.videoHeight;
  context.drawImage(elements.cameraVideo, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 0.92);
  });

  if (!blob) {
    showBanner("Khong the tao anh tu camera. Hay thu chup lai.", "error");
    return;
  }

  const file = new File([blob], `leaf-camera-${Date.now()}.jpg`, {
    type: "image/jpeg",
  });

  applyFile(file, {
    label: "Anh chup tu camera",
    fromCamera: true,
  });
  closeCameraPanel();
}

async function analyzeImage() {
  if (!state.file) {
    return;
  }

  setLoadingState(true);
  hideBanner();
  renderPipeline(basePipeline());

  const formData = new FormData();
  formData.append("image", state.file);
  formData.append("symptoms", (elements.symptomsInput?.value || "").trim());

  try {
    const response = await fetch(buildApiUrl("/api/analyze"), {
      method: "POST",
      body: formData,
    });
    const payload = await readJsonResponse(response, "Backend khong tra ve JSON hop le.");

    if (!response.ok || !payload.success) {
      throw new Error(payload.error || "Khong the phan tich anh.");
    }

    renderResult(payload.result);
  } catch (error) {
    showBanner(error.message, "error");
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  elements.analyzeButton.disabled = isLoading || !state.file;
  elements.analyzeButton.textContent = isLoading ? "Dang phan tich..." : "Phan tich ngay";
  elements.serverStatus.textContent = isLoading ? "Dang xu ly" : "San sang";
  elements.serverStatus.className = isLoading ? "status-pill warning" : "status-pill success";
}

function renderPipeline(items) {
  elements.pipelineList.innerHTML = items
    .map((item, index) => {
      const title = item.step || item.title;
      const detail = item.detail || "";
      const durationText = item.duration_ms ? `<br />Thoi gian: ${item.duration_ms} ms` : "";

      return `
        <article class="pipeline-item">
          <span class="step-index">${index + 1}</span>
          <div>
            <h3>${escapeHtml(title)}</h3>
            <p>${escapeHtml(detail)}${durationText}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderResult(result) {
  elements.resultsCard.classList.remove("hidden");
  elements.processingTime.textContent = `${result.meta.total_duration_ms} ms`;

  renderPipeline(result.pipeline);
  renderImages(result.images);
  renderClassification(result.classification);
  renderAdvice(result.llm);

  elements.resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderImages(images) {
  elements.resultOriginal.src = images.original || "";
  elements.resultAnnotated.src = images.annotated || "";
  elements.resultCrop.src = images.cropped_leaf || "";
}

function renderClassification(classification) {
  elements.cnnHeadline.textContent = `${classification.input_size.width} x ${classification.input_size.height}`;
  elements.cnnLabel.textContent = classification.display_label;
  elements.cnnConfidence.textContent = `Do tin cay: ${(classification.confidence * 100).toFixed(2)}%`;

  if (classification.warning) {
    elements.cnnWarning.textContent = classification.warning;
    elements.cnnWarning.classList.remove("hidden");
  } else {
    elements.cnnWarning.textContent = "";
    elements.cnnWarning.classList.add("hidden");
  }

  elements.predictionList.innerHTML = classification.top_predictions
    .map(
      (item) => `
        <div class="prediction-item">
          <div class="prediction-row">
            <strong>${escapeHtml(item.display_label)}</strong>
            <span>${(item.confidence * 100).toFixed(2)}%</span>
          </div>
          <div class="prediction-bar">
            <span style="width: ${(item.confidence * 100).toFixed(2)}%"></span>
          </div>
        </div>
      `
    )
    .join("");
}

function renderAdvice(llm) {
  elements.llmSource.textContent = `Nguon: ${llm.source} (${llm.model})`;
  elements.llmHeadline.textContent = llm.headline || "-";
  elements.llmSummary.textContent = llm.summary || "-";
  elements.llmWarning.textContent = llm.warning || "Khong co ghi chu them.";

  renderList(elements.careSteps, llm.care_steps);
  renderList(elements.nextSteps, llm.next_steps);
}

function renderList(target, items) {
  target.innerHTML = (items || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showBanner(message, type) {
  elements.messageBanner.textContent = message;
  elements.messageBanner.className = `message-banner ${type}`;
}

function hideBanner() {
  elements.messageBanner.textContent = "";
  elements.messageBanner.className = "message-banner hidden";
}

function buildApiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

async function readJsonResponse(response, fallbackMessage) {
  const contentType = (response.headers.get("content-type") || "").toLowerCase();
  const bodyText = await response.text();

  if (!contentType.includes("application/json")) {
    throw new Error(`${fallbackMessage} API dang tra ve ${contentType || "du lieu khong xac dinh"}.`);
  }

  try {
    return JSON.parse(bodyText);
  } catch {
    throw new Error(fallbackMessage);
  }
}

init();
