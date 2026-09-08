"use strict";

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const API_KEY_STORAGE = "fashion_api_key";
const SKELETON_COUNT = 8;

const COPY = {
  idle: 'Nhập mô tả sản phẩm để bắt đầu, vd: "váy đầm dự tiệc màu đỏ"',
  loading: "Đang tìm...",
  empty: "Không tìm thấy sản phẩm nào phù hợp. Thử mô tả khác hoặc bỏ lọc danh mục.",
  network: "Không kết nối được tới API. Kiểm tra xem uvicorn còn chạy không.",
  unauthorized: 'API key không đúng — nhập lại ở ô "API key".',
  tooLarge: "Ảnh quá lớn (tối đa 8 MB). Chọn ảnh nhỏ hơn.",
  invalid: "Truy vấn hoặc ảnh không hợp lệ.",
  notImplemented: "Hệ thống đang dùng adapter chưa cài đặt — kiểm tra FASHION_SEARCH_CONFIG.",
  unavailable: "Vector store (Qdrant) không phản hồi — hãy khởi động qdrant.exe rồi thử lại.",
  degraded: "Hệ thống suy giảm — Qdrant không trả lời, tìm kiếm sẽ lỗi.",
  healthUnreachable: "Không gọi được /health — API chưa chạy?",
  brokenImage: "ảnh không tải được",
  noQuery: "Nhập mô tả sản phẩm hoặc chọn một ảnh trước khi tìm.",
};

function imageUrl(path) {
  if (typeof path !== "string" || !path) return null;
  const segments = path.replace(/\\/g, "/").split("/").filter(Boolean);
  if (!segments.length) return null;
  return "/image/" + segments.map(encodeURIComponent).join("/");
}

function cardsFrom(body) {
  if (!body || typeof body !== "object") return [];
  const hits = body.hits;
  if (!Array.isArray(hits)) return [];
  const cards = [];
  hits.forEach((hit, index) => {
    if (!hit || typeof hit !== "object") return;
    const product = hit.product;
    if (!product || typeof product !== "object") return;
    const rank = Number.isFinite(hit.rank) ? hit.rank : index + 1;
    const score = Number.isFinite(hit.score) ? hit.score : null;
    const attributes = product.attributes;
    cards.push({
      rank: rank,
      score: score,
      productId: typeof product.product_id === "string" ? product.product_id : "",
      category: typeof product.category === "string" ? product.category : "",
      title: typeof product.title === "string" ? product.title : "",
      imageUrl: imageUrl(product.image_path),
      chips:
        attributes && typeof attributes === "object" && !Array.isArray(attributes)
          ? Object.values(attributes).filter((value) => typeof value === "string" && value.trim())
          : [],
    });
  });
  return cards;
}

function errorMessage(status) {
  if (status === 401 || status === 403) return COPY.unauthorized;
  if (status === 413) return COPY.tooLarge;
  if (status === 422) return COPY.invalid;
  if (status === 501) return COPY.notImplemented;
  if (status === 503) return COPY.unavailable;
  return `API trả về lỗi HTTP ${status}.`;
}

function detailText(body) {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (detail !== null && detail !== undefined) {
      try {
        return JSON.stringify(detail);
      } catch (err) {
        return "";
      }
    }
  }
  return "";
}

function healthPill(payload) {
  if (!payload || typeof payload !== "object") {
    return { level: "bad", text: COPY.healthUnreachable };
  }
  const retriever = typeof payload.retriever === "string" ? payload.retriever : "?";
  const count = Number.isFinite(payload.indexed_products) ? payload.indexed_products : 0;
  if (payload.status === "ok") {
    return { level: "ok", text: `${retriever} · ${count} sản phẩm đã index` };
  }
  if (payload.status === "degraded") {
    return { level: "warn", text: COPY.degraded };
  }
  return { level: "warn", text: `Trạng thái lạ "${payload.status}" · ${retriever}` };
}

const el = {
  form: document.getElementById("search-form"),
  query: document.getElementById("q"),
  topK: document.getElementById("top-k"),
  category: document.getElementById("category"),
  apiKey: document.getElementById("api-key"),
  submit: document.getElementById("submit"),
  dropZone: document.getElementById("drop-zone"),
  file: document.getElementById("file"),
  clearFile: document.getElementById("clear-file"),
  previewWrap: document.getElementById("preview-wrap"),
  preview: document.getElementById("preview"),
  previewName: document.getElementById("preview-name"),
  feedback: document.getElementById("feedback"),
  grid: document.getElementById("grid"),
  status: document.getElementById("status"),
};

let selectedFile = null;
let previewObjectUrl = null;
let inFlight = null;

function setStatus(level, text) {
  el.status.className = `pill ${level}`;
  el.status.textContent = text;
}

function showMessage(text, level, detail) {
  const box = document.createElement("div");
  box.className = `message ${level}`;
  const primary = document.createElement("div");
  primary.textContent = text;
  box.appendChild(primary);
  if (detail) {
    const small = document.createElement("div");
    small.className = "detail";
    small.textContent = detail;
    box.appendChild(small);
  }
  el.feedback.replaceChildren(box);
}

function clearMessage() {
  el.feedback.replaceChildren();
}

function renderSkeletons() {
  const frame = document.createDocumentFragment();
  for (let i = 0; i < SKELETON_COUNT; i += 1) {
    const card = document.createElement("div");
    card.className = "card skeleton";
    const thumb = document.createElement("div");
    thumb.className = "shimmer thumb";
    const line = document.createElement("div");
    line.className = "shimmer line";
    const short = document.createElement("div");
    short.className = "shimmer line short";
    card.append(thumb, line, short);
    frame.appendChild(card);
  }
  el.grid.replaceChildren(frame);
}

function buildCard(card) {
  const node = document.createElement("div");
  node.className = "card";

  if (card.imageUrl) {
    const img = document.createElement("img");
    img.src = card.imageUrl;
    img.loading = "lazy";
    img.alt = "";
    img.addEventListener("error", () => {
      img.remove();
      const fallback = document.createElement("div");
      fallback.className = "thumb blank";
      const note = document.createElement("div");
      note.className = "broken";
      note.textContent = COPY.brokenImage;
      node.prepend(note);
      node.prepend(fallback);
    });
    node.appendChild(img);
  }

  const badge = document.createElement("span");
  badge.className = "rank";
  badge.textContent = `#${card.rank}`;
  node.appendChild(badge);

  const meta = document.createElement("div");
  meta.className = "meta";

  const headline = document.createElement("div");
  headline.className = "headline";
  headline.textContent = card.category || card.productId || "(không rõ danh mục)";
  meta.appendChild(headline);

  if (card.title) {
    const secondary = document.createElement("div");
    secondary.className = "secondary";
    secondary.textContent = card.title;
    meta.appendChild(secondary);
  }

  const score = document.createElement("div");
  score.className = "score";
  const parts = [];
  if (card.productId) parts.push(card.productId);
  if (card.score !== null) parts.push(card.score.toFixed(3));
  score.textContent = parts.join(" · ");
  if (parts.length) meta.appendChild(score);

  if (card.chips.length) {
    const chips = document.createElement("div");
    chips.className = "chips";
    card.chips.forEach((value) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = value;
      chips.appendChild(chip);
    });
    meta.appendChild(chips);
  }

  node.appendChild(meta);
  return node;
}

function renderCards(cards) {
  const frame = document.createDocumentFragment();
  cards.forEach((card) => frame.appendChild(buildCard(card)));
  el.grid.replaceChildren(frame);
}

function setBusy(busy) {
  el.submit.disabled = busy;
  el.submit.textContent = busy ? COPY.loading : "Tìm";
  [el.query, el.topK, el.category, el.apiKey, el.file, el.clearFile].forEach((input) => {
    input.disabled = busy;
  });
}

async function refreshStatus() {
  let payload = null;
  try {
    const response = await fetch("/health");
    payload = response.ok ? await response.json() : null;
  } catch (err) {
    payload = null;
  }
  const pill = healthPill(payload);
  setStatus(pill.level, pill.text);
}

function apiKey() {
  return el.apiKey.value.trim();
}

function authHeaders() {
  const key = apiKey();
  return key ? { "X-API-Key": key } : {};
}

function clearFileSelection() {
  selectedFile = null;
  el.file.value = "";
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
  el.preview.removeAttribute("src");
  el.previewName.textContent = "";
  el.previewWrap.hidden = true;
  el.clearFile.hidden = true;
}

function selectFile(file) {
  if (!file) return;
  if (file.size > MAX_IMAGE_BYTES) {
    clearFileSelection();
    showMessage(COPY.tooLarge, "bad");
    return;
  }
  clearMessage();
  selectedFile = file;
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = URL.createObjectURL(file);
  el.preview.src = previewObjectUrl;
  el.previewName.textContent = file.name;
  el.previewWrap.hidden = false;
  el.clearFile.hidden = false;
}

function topKValue() {
  const value = Number.parseInt(el.topK.value, 10);
  if (!Number.isFinite(value)) return 12;
  return Math.min(100, Math.max(1, value));
}

function buildRequest(signal) {
  const category = el.category.value.trim();
  if (selectedFile) {
    const form = new FormData();
    form.append("image", selectedFile, selectedFile.name || "query.jpg");
    form.append("top_k", String(topKValue()));
    if (category) form.append("category", category);
    return ["/search/image", { method: "POST", headers: authHeaders(), body: form, signal: signal }];
  }
  const payload = { text: el.query.value.trim(), top_k: topKValue() };
  if (category) payload.filters = { category: category };
  return [
    "/search",
    {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
      body: JSON.stringify(payload),
      signal: signal,
    },
  ];
}

async function runSearch(event) {
  event.preventDefault();
  if (!selectedFile && !el.query.value.trim()) {
    showMessage(COPY.noQuery, "info");
    return;
  }

  if (inFlight) inFlight.abort();
  const controller = new AbortController();
  inFlight = controller;

  sessionStorage.setItem(API_KEY_STORAGE, apiKey());
  clearMessage();
  renderSkeletons();
  setBusy(true);

  const [url, options] = buildRequest(controller.signal);
  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    if (controller.signal.aborted) return;
    el.grid.replaceChildren();
    showMessage(COPY.network, "bad");
    setBusy(false);
    inFlight = null;
    refreshStatus();
    return;
  }

  let body = null;
  try {
    body = await response.json();
  } catch (err) {
    body = null;
  }

  if (controller.signal.aborted) return;
  inFlight = null;
  setBusy(false);

  if (!response.ok) {
    el.grid.replaceChildren();
    showMessage(errorMessage(response.status), "bad", detailText(body));
    refreshStatus();
    return;
  }

  const cards = cardsFrom(body);
  if (!cards.length) {
    el.grid.replaceChildren();
    showMessage(COPY.empty, "info");
    return;
  }
  renderCards(cards);
}

el.apiKey.value = sessionStorage.getItem(API_KEY_STORAGE) || "";
el.form.addEventListener("submit", runSearch);
el.file.addEventListener("change", () => selectFile(el.file.files[0]));
el.clearFile.addEventListener("click", clearFileSelection);

["dragenter", "dragover"].forEach((name) => {
  el.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    el.dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((name) => {
  el.dropZone.addEventListener(name, () => el.dropZone.classList.remove("dragging"));
});
el.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  const files = event.dataTransfer && event.dataTransfer.files;
  if (files && files.length) selectFile(files[0]);
});

showMessage(COPY.idle, "info");
refreshStatus();
