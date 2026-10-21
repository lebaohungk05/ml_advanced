"use strict";

(function () {
  const STORAGE_KEY = "fashion_lang";
  const root = document.documentElement;

  const DICT = {
    vi: {
      title: "Tìm kiếm thời trang đa mô thức",
      skip: "Bỏ qua tới kết quả",
      status_checking: "Đang kiểm tra hệ thống...",
      theme_to_dark: "Nền tối",
      theme_to_light: "Nền sáng",
      lang_switch: "English",
      lang_switch_aria: "Switch to English",
      hero_a: "Món đồ bạn",
      hero_b: "đang nghĩ tới",
      subtitle: "Mô tả bằng tiếng Việt hoặc thả một tấm ảnh. Hệ thống tìm những sản phẩm gần nhất trong catalog.",
      search_label: "Mô tả sản phẩm",
      q_placeholder: "váy đầm dự tiệc màu đỏ",
      submit: "Tìm",
      rail_label: "Chọn nhanh danh mục",
      cat_dress: "Váy đầm",
      cat_skirt: "Chân váy",
      cat_shirt: "Sơ mi",
      cat_jacket: "Áo khoác",
      cat_trousers: "Quần dài",
      cat_accessory: "Phụ kiện",
      cat_all: "Tất cả",
      category: "Danh mục",
      category_placeholder: "Tất cả danh mục",
      top_k: "Số kết quả",
      api_key: "API key",
      api_placeholder: "Nếu server bật xác thực",
      drop_aria: "Tìm bằng ảnh",
      drop_title: "Có sẵn một tấm ảnh?",
      drop_text: "Kéo ảnh vào đây hoặc chọn tệp. Khi đã chọn ảnh, nút “Tìm” sẽ tìm theo ảnh thay vì theo văn bản.",
      file_aria: "Chọn ảnh truy vấn",
      clear_file: "Bỏ ảnh",
      preview_alt: "Ảnh truy vấn",
      results: "Kết quả tìm kiếm",
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
      indexed: "{retriever} · {count} sản phẩm đã index",
      strange: 'Trạng thái lạ "{status}" · {retriever}',
      http_error: "API trả về lỗi HTTP {status}.",
      unknown_category: "(không rõ danh mục)",
    },
    en: {
      title: "Multimodal Fashion Search",
      skip: "Skip to results",
      status_checking: "Checking system...",
      theme_to_dark: "Dark mode",
      theme_to_light: "Light mode",
      lang_switch: "Tiếng Việt",
      lang_switch_aria: "Chuyển sang tiếng Việt",
      hero_a: "Find your",
      hero_b: "next look",
      subtitle: "Describe it in words or drop in a photo. The system finds the closest products in the catalog.",
      search_label: "Product description",
      q_placeholder: "red evening dress",
      submit: "Search",
      rail_label: "Quick category filter",
      cat_dress: "Dresses",
      cat_skirt: "Skirts",
      cat_shirt: "Shirts",
      cat_jacket: "Jackets",
      cat_trousers: "Trousers",
      cat_accessory: "Accessories",
      cat_all: "All",
      category: "Category",
      category_placeholder: "All categories",
      top_k: "Results",
      api_key: "API key",
      api_placeholder: "If the server requires auth",
      drop_aria: "Search by image",
      drop_title: "Got a photo already?",
      drop_text: "Drag an image here or choose a file. Once an image is selected, “Search” looks up by image instead of text.",
      file_aria: "Choose a query image",
      clear_file: "Remove image",
      preview_alt: "Query image",
      results: "Search results",
      idle: 'Describe a product to get started, e.g. "red evening dress"',
      loading: "Searching...",
      empty: "No matching products found. Try a different description or clear the category filter.",
      network: "Cannot reach the API. Check that uvicorn is still running.",
      unauthorized: 'Wrong API key — enter it again in the "API key" field.',
      tooLarge: "Image too large (8 MB max). Pick a smaller one.",
      invalid: "Invalid query or image.",
      notImplemented: "The system is using an unimplemented adapter — check FASHION_SEARCH_CONFIG.",
      unavailable: "The vector store (Qdrant) is not responding — start qdrant.exe and try again.",
      degraded: "System degraded — Qdrant is not answering, searches will fail.",
      healthUnreachable: "Cannot reach /health — is the API running?",
      brokenImage: "image failed to load",
      noQuery: "Enter a product description or choose an image before searching.",
      indexed: "{retriever} · {count} products indexed",
      strange: 'Unexpected status "{status}" · {retriever}',
      http_error: "The API returned HTTP error {status}.",
      unknown_category: "(unknown category)",
    },
  };

  function stored() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (err) {
      return null;
    }
  }

  function initialLang() {
    const saved = stored();
    if (saved === "vi" || saved === "en") return saved;
    return (navigator.language || "vi").toLowerCase().startsWith("vi") ? "vi" : "en";
  }

  let current = initialLang();

  function t(key, vars) {
    const table = DICT[current];
    let text = Object.prototype.hasOwnProperty.call(table, key) ? table[key] : key;
    if (vars) {
      Object.entries(vars).forEach(([name, value]) => {
        text = text.split(`{${name}}`).join(String(value));
      });
    }
    return text;
  }

  function themeLabelKey() {
    return root.getAttribute("data-theme") === "dark" ? "theme_to_light" : "theme_to_dark";
  }

  function translate() {
    root.setAttribute("lang", current);
    document.title = t("title");
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
      node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder));
    });
    document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
      node.setAttribute("aria-label", t(node.dataset.i18nAria));
    });
    document.querySelectorAll("[data-i18n-alt]").forEach((node) => {
      node.setAttribute("alt", t(node.dataset.i18nAlt));
    });
    const themeLabel = document.querySelector("#theme-toggle .theme-label");
    if (themeLabel) themeLabel.textContent = t(themeLabelKey());
    const langToggle = document.getElementById("lang-toggle");
    if (langToggle) {
      langToggle.querySelector(".lang-label").textContent = t("lang_switch");
      langToggle.setAttribute("aria-label", t("lang_switch_aria"));
    }
  }

  function setLang(next) {
    current = next;
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch (err) {
      current = next;
    }
    translate();
    document.dispatchEvent(new CustomEvent("langchange", { detail: next }));
  }

  window.I18N = { t: t, refresh: translate, lang: () => current };

  document.addEventListener("DOMContentLoaded", () => {
    translate();
    document.getElementById("lang-toggle").addEventListener("click", () => {
      setLang(current === "vi" ? "en" : "vi");
    });
  });
})();
