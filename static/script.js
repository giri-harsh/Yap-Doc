(() => {
  const sessionId = (() => {
    let id = localStorage.getItem("yapdoc_session");
    if (!id) {
      id = (crypto.randomUUID
        ? crypto.randomUUID()
        : `s-${Date.now()}-${Math.random().toString(16).slice(2)}`);
      localStorage.setItem("yapdoc_session", id);
    }
    return id;
  })();

  const stage = document.getElementById("stage");
  const chat = document.getElementById("chat");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const fileChip = document.getElementById("fileChip");
  const fcName = document.getElementById("fcName");
  const fcMeta = document.getElementById("fcMeta");
  const fcRemove = document.getElementById("fcRemove");
  const modelPicker = document.getElementById("modelPicker");
  const deepToggle = document.getElementById("deepToggle");
  const promptInput = document.getElementById("promptInput");
  const sendBtn = document.getElementById("sendBtn");
  const homeBtn = document.getElementById("homeBtn");
  const deepToast = document.getElementById("deepToast");
  const toastClose = document.getElementById("toastClose");

  let modelChoice = "Groq";
  let messages = []; // {role, content}
  let toastTimer = null;

  // ── Model picker ──────────────────────────────────────
  modelPicker.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    modelPicker.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    modelChoice = btn.dataset.model;
  });

  // ── Deep dive toggle + toast ───────────────────────────
  function showDeepToast() {
    deepToast.hidden = false;
    requestAnimationFrame(() => deepToast.classList.add("show"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(hideDeepToast, 5000);
  }
  function hideDeepToast() {
    deepToast.classList.remove("show");
    clearTimeout(toastTimer);
    setTimeout(() => { deepToast.hidden = true; }, 200);
  }
  deepToggle.addEventListener("change", () => {
    if (deepToggle.checked) showDeepToast();
    else hideDeepToast();
  });
  toastClose.addEventListener("click", hideDeepToast);

  // ── Home / back button — start a new conversation ─────
  homeBtn.addEventListener("click", async () => {
    messages = [];
    chat.innerHTML = "";
    stage.classList.remove("has-messages");
    document.body.classList.remove("in-chat");

    fileChip.hidden = true;
    dropzone.hidden = false;
    fileInput.value = "";
    await fetch("/api/document", { method: "DELETE", headers: { "X-Session-Id": sessionId } });
  });

  // ── Dropzone ──────────────────────────────────────────
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  async function handleFile(file) {
    const allowed = ["pdf", "docx", "txt"];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
      alert("Please upload a PDF, DOCX, or TXT file.");
      return;
    }

    const label = dropzone.querySelector(".dz-text strong");
    const originalLabel = label.textContent;
    label.textContent = "Reading document…";
    dropzone.style.pointerEvents = "none";

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        headers: { "X-Session-Id": sessionId },
        body: form,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      fcName.textContent = data.filename;
      fcMeta.textContent = `${data.chunks} chunks`;
      fileChip.hidden = false;
      dropzone.hidden = true;
    } catch (err) {
      alert(`Couldn't read that document: ${err.message}`);
    } finally {
      label.textContent = originalLabel;
      dropzone.style.pointerEvents = "";
    }
  }

  fcRemove.addEventListener("click", async () => {
    await fetch("/api/document", { method: "DELETE", headers: { "X-Session-Id": sessionId } });
    fileChip.hidden = true;
    dropzone.hidden = false;
    fileInput.value = "";
  });

  // ── Composer ──────────────────────────────────────────
  function autoGrow() {
    promptInput.style.height = "auto";
    promptInput.style.height = Math.min(promptInput.scrollHeight, 140) + "px";
  }
  promptInput.addEventListener("input", () => {
    autoGrow();
    sendBtn.disabled = !promptInput.value.trim();
  });
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  sendBtn.disabled = true;
  sendBtn.addEventListener("click", send);

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function addUserMessage(text) {
    const wrap = document.createElement("div");
    wrap.className = "msg user";
    const bubble = document.createElement("div");
    bubble.className = "bubble-user";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    stage.classList.add("has-messages");
    document.body.classList.add("in-chat");
    chat.scrollTop = chat.scrollHeight;
  }

  function addAssistantMessage(markdown) {
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    const html = window.marked ? marked.parse(markdown) : `<p>${escapeHtml(markdown)}</p>`;
    wrap.innerHTML = `<div class="card-assistant"><span class="card-label">Yap-Doc</span>${html}</div>`;
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
  }

  function addYappingIndicator() {
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    wrap.id = "yapping-indicator";
    wrap.innerHTML = `
      <div class="yapping">
        <span class="wave"><i></i><i></i><i></i><i></i></span>
        yapping…
      </div>`;
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
  }
  function removeYappingIndicator() {
    document.getElementById("yapping-indicator")?.remove();
  }

  async function send() {
    const text = promptInput.value.trim();
    if (!text) return;

    addUserMessage(text);
    const historyForApi = messages.slice(-6);
    messages.push({ role: "user", content: text });

    promptInput.value = "";
    autoGrow();
    sendBtn.disabled = true;
    addYappingIndicator();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Session-Id": sessionId },
        body: JSON.stringify({
          query: text,
          model_choice: modelChoice,
          deep_research: deepToggle.checked,
          history: historyForApi,
        }),
      });
      const data = await res.json();
      removeYappingIndicator();
      const answer = data.answer || "Hmm, I didn't get a response back. Try again?";
      addAssistantMessage(answer);
      messages.push({ role: "assistant", content: answer });
    } catch (err) {
      removeYappingIndicator();
      addAssistantMessage(`⚠️ Something went wrong reaching the server: \`${err.message}\``);
    }
  }
})();
