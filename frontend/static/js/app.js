const state = {
  file: null,
  previewUrl: "",
  mediaStream: null,
};

const API_BASE_URL = String(window.APP_CONFIG?.API_BASE_URL || window.location.origin).replace(/\/$/, "");

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
  chatToggle: document.getElementById("chatToggle"),
  chatPanel: document.getElementById("chatPanel"),
  chatClose: document.getElementById("chatClose"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  chatMessages: document.getElementById("chatMessages"),
  chatSend: document.getElementById("chatSend"),
};

const CAMERA_DEFAULT_TEXT = "Camera sẽ hoạt động tốt trên localhost và bản Render đã bật HTTPS.";

function basePipeline() {
  return [
    { title: "YOLO nhận diện lá", detail: "Tách vùng lá rõ nhất trước khi đưa sang CNN." },
    { title: "CNN phân loại", detail: "Đọc ảnh crop và tính xác suất cho từng lớp của model_0.h5." },
    { title: "AI tư vấn", detail: "Tóm tắt ngắn gọn, dễ hiểu và gợi ý chăm sóc tiếp theo." },
  ];
}

async function init() {
  renderPipeline(basePipeline());
  bindEvents();
  syncCameraAvailability();
  await loadHealth();
}

function bindEvents() {
  elements.fileInput.addEventListener("change", (event) => applyFile(event.target.files[0]));
  elements.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("drag-over");
  });
  elements.dropzone.addEventListener("dragleave", () => elements.dropzone.classList.remove("drag-over"));
  elements.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("drag-over");
    applyFile(event.dataTransfer.files[0]);
  });

  elements.openCameraButton.addEventListener("click", startCamera);
  elements.captureButton.addEventListener("click", captureImage);
  elements.closeCameraButton.addEventListener("click", closeCameraPanel);
  elements.analyzeButton.addEventListener("click", analyzeImage);
  elements.resetButton.addEventListener("click", resetForm);
  elements.chatToggle.addEventListener("click", openChat);
  elements.chatClose.addEventListener("click", closeChat);
  elements.chatForm.addEventListener("submit", sendChatMessage);
  window.addEventListener("beforeunload", stopCameraStream);
}

function syncCameraAvailability() {
  if (hasCameraSupport()) return;
  elements.openCameraButton.disabled = true;
  elements.cameraHelper.textContent = "Trình duyệt hiện tại không hỗ trợ chụp ảnh trực tiếp. Bạn vẫn có thể tải ảnh thủ công.";
}

function hasCameraSupport() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function applyFile(file, options = {}) {
  if (!file) return;
  if (file.type && !file.type.startsWith("image/")) {
    showBanner("Vui lòng chọn đúng file ảnh JPG, PNG hoặc WEBP.", "error");
    return;
  }

  state.file = file;
  elements.fileName.textContent = options.label || file.name;
  elements.analyzeButton.disabled = false;

  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = state.previewUrl;
  elements.previewImage.classList.remove("is-empty");

  if (options.fromCamera) showBanner("Đã chụp ảnh thành công. Bạn có thể bấm Phân tích ngay.", "info");
}

function resetForm() {
  state.file = null;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = "";
  elements.fileInput.value = "";
  elements.symptomsInput.value = "";
  elements.fileName.textContent = "Chưa chọn ảnh";
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
  elements.cnnHeadline.textContent = "Chưa có dữ liệu";
  elements.cnnLabel.textContent = "-";
  elements.cnnConfidence.textContent = "Độ tin cậy: -";
  elements.cnnWarning.textContent = "";
  elements.cnnWarning.classList.add("hidden");
  elements.predictionList.innerHTML = "";
  elements.llmSource.textContent = "Nguồn: -";
  elements.llmHeadline.textContent = "-";
  elements.llmSummary.textContent = "-";
  elements.careSteps.innerHTML = "";
  elements.nextSteps.innerHTML = "";
  elements.llmWarning.textContent = "-";
}

async function loadHealth() {
  try {
    const response = await fetch(buildApiUrl("/api/health"));
    const data = await readJsonResponse(response, "Không đọc được trạng thái backend.");
    const dependencies = data.dependencies;

    elements.serverStatus.textContent = "Sẵn sàng";
    elements.serverStatus.className = "status-pill success";
    elements.healthYolo.textContent = dependencies.yolo_model_found && dependencies.ultralytics_ready ? "Sẵn sàng" : dependencies.yolo_model_found ? "Thiếu ultralytics" : "Thiếu model";
    elements.healthCnn.textContent = dependencies.cnn_model_found && dependencies.tensorflow_ready ? "Sẵn sàng" : dependencies.cnn_model_found ? "Thiếu TensorFlow" : "Thiếu model";
    elements.healthLabels.textContent = dependencies.cnn_labels_found ? "Có file nhãn" : "Đang dùng nhãn mẫu";
    elements.healthOpenAi.textContent = dependencies.openai_key_configured ? "Đã cấu hình" : "Chưa có API key";

    if (!dependencies.ultralytics_ready || !dependencies.tensorflow_ready) {
      showBanner("Môi trường hiện tại đang thiếu một số thư viện ML. Website vẫn mở được, nhưng suy luận model có thể không đầy đủ.", "info");
    }
  } catch (error) {
    elements.serverStatus.textContent = "Không kết nối";
    elements.serverStatus.className = "status-pill warning";
    elements.healthYolo.textContent = "Không rõ";
    elements.healthCnn.textContent = "Không rõ";
    elements.healthLabels.textContent = "Không rõ";
    elements.healthOpenAi.textContent = "Không rõ";
  }
}

async function startCamera() {
  if (!hasCameraSupport()) {
    showBanner("Trình duyệt không hỗ trợ camera trực tiếp.", "error");
    return;
  }
  hideBanner();
  elements.cameraShell.classList.remove("hidden");
  elements.cameraHelper.textContent = "Đang yêu cầu quyền camera...";
  stopCameraStream();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
    state.mediaStream = stream;
    elements.cameraVideo.srcObject = stream;
    await elements.cameraVideo.play();
    elements.openCameraButton.classList.add("hidden");
    elements.captureButton.classList.remove("hidden");
    elements.closeCameraButton.classList.remove("hidden");
    elements.cameraHelper.textContent = "Camera đã sẵn sàng. Căn lá vào giữa khung rồi bấm Chụp ảnh.";
  } catch (error) {
    closeCameraPanel();
    showBanner("Không mở được camera. Hãy cấp quyền camera cho trình duyệt và thử lại.", "error");
  }
}

function stopCameraStream() {
  if (!state.mediaStream) return;
  for (const track of state.mediaStream.getTracks()) track.stop();
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
    showBanner("Camera chưa sẵn sàng để chụp. Hãy đợi một chút rồi thử lại.", "error");
    return;
  }
  const canvas = elements.captureCanvas;
  const context = canvas.getContext("2d");
  canvas.width = elements.cameraVideo.videoWidth;
  canvas.height = elements.cameraVideo.videoHeight;
  context.drawImage(elements.cameraVideo, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
  if (!blob) {
    showBanner("Không thể tạo ảnh từ camera. Hãy thử chụp lại.", "error");
    return;
  }
  applyFile(new File([blob], `leaf-camera-${Date.now()}.jpg`, { type: "image/jpeg" }), {
    label: "Ảnh chụp từ camera",
    fromCamera: true,
  });
  closeCameraPanel();
}

async function analyzeImage() {
  if (!state.file) return;
  setLoadingState(true);
  hideBanner();
  renderPipeline(basePipeline());

  const formData = new FormData();
  formData.append("image", state.file);
  formData.append("symptoms", (elements.symptomsInput?.value || "").trim());

  try {
    const response = await fetch(buildApiUrl("/api/analyze"), { method: "POST", body: formData });
    const payload = await readJsonResponse(response, "Backend không trả về JSON hợp lệ.");
    if (!response.ok || !payload.success) throw new Error(payload.error || "Không thể phân tích ảnh.");
    renderResult(payload.result);
  } catch (error) {
    showBanner(error.message, "error");
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  elements.analyzeButton.disabled = isLoading || !state.file;
  elements.analyzeButton.textContent = isLoading ? "Đang phân tích..." : "Phân tích ngay";
  elements.serverStatus.textContent = isLoading ? "Đang xử lý" : "Sẵn sàng";
  elements.serverStatus.className = isLoading ? "status-pill warning" : "status-pill success";
}

function renderPipeline(items) {
  elements.pipelineList.innerHTML = items
    .map((item, index) => {
      const title = item.step || item.title;
      const detail = item.detail || "";
      const durationText = item.duration_ms ? `<br />Thời gian: ${item.duration_ms} ms` : "";
      return `<article class="pipeline-item"><span class="step-index">${index + 1}</span><div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(detail)}${durationText}</p></div></article>`;
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
  elements.cnnConfidence.textContent = `Độ tin cậy: ${(classification.confidence * 100).toFixed(2)}%`;
  elements.cnnWarning.textContent = classification.warning || "";
  elements.cnnWarning.classList.toggle("hidden", !classification.warning);
  elements.predictionList.innerHTML = classification.top_predictions
    .map((item) => `<div class="prediction-item"><div class="prediction-row"><strong>${escapeHtml(item.display_label)}</strong><span>${(item.confidence * 100).toFixed(2)}%</span></div><div class="prediction-bar"><span style="width: ${(item.confidence * 100).toFixed(2)}%"></span></div></div>`)
    .join("");
}

function renderAdvice(llm) {
  elements.llmSource.textContent = `Nguồn: ${llm.source} (${llm.model})`;
  elements.llmHeadline.textContent = llm.headline || "-";
  elements.llmSummary.textContent = llm.summary || "-";
  elements.llmWarning.textContent = llm.warning || "Không có ghi chú thêm.";
  renderList(elements.careSteps, llm.care_steps);
  renderList(elements.nextSteps, llm.next_steps);
}

function renderList(target, items) {
  target.innerHTML = (items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function openChat() {
  elements.chatPanel.classList.remove("hidden");
  elements.chatToggle.classList.add("hidden");
  elements.chatInput.focus();
}

function closeChat() {
  elements.chatPanel.classList.add("hidden");
  elements.chatToggle.classList.remove("hidden");
}

async function sendChatMessage(event) {
  event.preventDefault();
  const message = elements.chatInput.value.trim();
  if (!message) return;

  addChatMessage(message, "user");
  elements.chatInput.value = "";
  elements.chatSend.disabled = true;
  const typing = addChatMessage("Chuyên gia đang trả lời...", "bot muted");

  try {
    const response = await fetch(buildApiUrl("/api/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const payload = await readJsonResponse(response, "Backend không trả về phản hồi chat hợp lệ.");
    if (!response.ok || !payload.success) throw new Error(payload.error || "Không gửi được câu hỏi.");
    typing.textContent = payload.result.reply || "Chuyên gia chưa có phản hồi.";
  } catch (error) {
    typing.textContent = error.message;
  } finally {
    elements.chatSend.disabled = false;
  }
}

function addChatMessage(text, type) {
  const message = document.createElement("div");
  message.className = `chat-message ${type}`;
  message.textContent = text;
  elements.chatMessages.appendChild(message);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  return message;
}

function escapeHtml(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
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
    throw new Error(`${fallbackMessage} API đang trả về ${contentType || "dữ liệu không xác định"}.`);
  }
  try {
    return JSON.parse(bodyText);
  } catch {
    throw new Error(fallbackMessage);
  }
}

init();
