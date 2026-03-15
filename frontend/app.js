/**
 * Coragem AI — Test Frontend
 *
 * Vanilla JS client for signup, login, document upload, and RAG chat.
 * Connects to the FastAPI backend at the same origin (served via /app).
 */

const API = window.location.origin;
let token = localStorage.getItem("token") || null;
let currentUser = null;

// ── Helpers ──────────────────────────────────────────────────────────────────

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (!(opts.body instanceof FormData)) headers["Content-Type"] = "application/json";

  const res = await fetch(`${API}${path}`, { ...opts, headers });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const msg = data?.message || data?.detail || `Error ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

// ── Auth ─────────────────────────────────────────────────────────────────────

// Tab switching
$$(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.tab;
    if (target === "login") {
      show($("#login-form"));
      hide($("#register-form"));
    } else {
      hide($("#login-form"));
      show($("#register-form"));
    }
  });
});

// Login
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#login-error").textContent = "";

  try {
    const res = await api("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({
        email: $("#login-email").value,
        password: $("#login-password").value,
      }),
    });

    token = res.data.access_token;
    localStorage.setItem("token", token);
    currentUser = res.data;
    enterApp();
  } catch (err) {
    $("#login-error").textContent = err.message;
  }
});

// Register
$("#register-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#register-error").textContent = "";
  $("#register-success").textContent = "";

  try {
    await api("/api/v1/auth/register", {
      method: "POST",
      body: JSON.stringify({
        full_name: $("#reg-name").value,
        email: $("#reg-email").value,
        password: $("#reg-password").value,
        department: $("#reg-department").value,
        role: $("#reg-role").value,
      }),
    });

    $("#register-success").textContent = "Account created! Switch to Login.";
    $("#register-form").reset();
  } catch (err) {
    $("#register-error").textContent = err.message;
  }
});

// Logout
$("#logout-btn").addEventListener("click", () => {
  token = null;
  currentUser = null;
  localStorage.removeItem("token");
  $("#auth-screen").classList.add("active");
  $("#app-screen").classList.remove("active");
  $("#messages").innerHTML = "";
});

// ── App Navigation ──────────────────────────────────────────────────────────

$$(".tab-nav").forEach(btn => {
  btn.addEventListener("click", () => {
    $$(".tab-nav").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    $$(".view").forEach(v => v.classList.remove("active"));
    $(`#${btn.dataset.view}-view`).classList.add("active");
  });
});

async function enterApp() {
  // Fetch user profile if we don't have it
  if (!currentUser) {
    try {
      const res = await api("/api/v1/auth/me");
      currentUser = res.data;
    } catch {
      // Token expired
      token = null;
      localStorage.removeItem("token");
      return;
    }
  }

  $("#user-badge").textContent = `${currentUser.department} - ${currentUser.full_name || currentUser.email}`;
  $("#auth-screen").classList.remove("active");
  $("#app-screen").classList.add("active");

  // Add welcome message
  if ($("#messages").children.length === 0) {
    addMessage("system", `Welcome! Ask me anything about your company documents.`);
  }
}

// ── Chat ─────────────────────────────────────────────────────────────────────

let sessionId = crypto.randomUUID();

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = renderMarkdown(content);
  $("#messages").appendChild(div);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return div;
}

function renderMarkdown(text) {
  // Minimal markdown: bold, lists, line breaks
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/^\s*[-*]\s+(.+)/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
    .replace(/\n/g, "<br>");
}

async function sendQuery() {
  const input = $("#query-input");
  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  addMessage("user", query);
  hide($("#sources-panel"));

  const assistantDiv = addMessage("assistant", "");
  assistantDiv.innerHTML = '<span class="spinner"></span> Thinking...';

  try {
    const res = await fetch(`${API}/api/v1/rag/query/stream`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        max_results: 5,
        confidence_threshold: 0.5,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.message || `Error ${res.status}`);
    }

    // Parse SSE stream
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullAnswer = "";
    let firstChunk = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // Keep incomplete line

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          var eventType = line.slice(7).trim();
        }
        if (line.startsWith("data: ")) {
          const data = line.slice(6).trim();
          if (!data) continue;

          try {
            const parsed = JSON.parse(data);

            if (eventType === "sources") {
              showSources(parsed);
            } else if (eventType === "message") {
              if (firstChunk) {
                assistantDiv.innerHTML = "";
                firstChunk = false;
              }
              fullAnswer += parsed.content;
              assistantDiv.innerHTML = renderMarkdown(fullAnswer);
              $("#messages").scrollTop = $("#messages").scrollHeight;
            } else if (eventType === "error") {
              assistantDiv.innerHTML = `<span style="color:var(--error)">${parsed.error}</span>`;
            }
          } catch {}
        }
      }
    }

    if (firstChunk) {
      // No message chunks received
      assistantDiv.innerHTML = "No response received.";
    }

  } catch (err) {
    assistantDiv.innerHTML = `<span style="color:var(--error)">${err.message}</span>`;
  }
}

function showSources(sources) {
  if (!sources || sources.length === 0) {
    hide($("#sources-panel"));
    return;
  }

  const list = $("#sources-list");
  list.innerHTML = sources.map(s =>
    `<span class="source-chip">${s.title} <span class="score">${(s.relevance_score * 100).toFixed(0)}%</span></span>`
  ).join("");
  show($("#sources-panel"));
}

$("#send-btn").addEventListener("click", sendQuery);
$("#query-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

// ── Upload ───────────────────────────────────────────────────────────────────

// File input display
$("#doc-file").addEventListener("change", (e) => {
  const name = e.target.files[0]?.name || "No file selected";
  $("#file-name").textContent = name;
});

// Click file wrapper to trigger file input
$(".file-input-wrapper").addEventListener("click", () => {
  $("#doc-file").click();
});

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = $("#doc-file").files[0];
  if (!file) return;

  const form = new FormData();
  form.append("title", $("#doc-title").value);
  form.append("department", $("#doc-department").value);
  form.append("doc_type", $("#doc-type").value);
  form.append("file", file);

  const allowedDepts = $("#doc-allowed-depts").value.trim();
  if (allowedDepts) form.append("allowed_departments", allowedDepts);

  show($("#upload-status"));
  hide($("#upload-result"));
  $("#upload-message").textContent = "Uploading and processing...";

  try {
    const res = await fetch(`${API}/api/v1/documents/ingest/pdf`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${token}` },
      body: form,
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.message || "Upload failed");

    // Start polling task status
    $("#upload-message").textContent = "Processing document...";
    await pollTaskStatus(data.task_id);

  } catch (err) {
    hide($("#upload-status"));
    const result = $("#upload-result");
    result.className = "error";
    result.textContent = err.message;
    show(result);
  }
});

async function pollTaskStatus(taskId) {
  const maxAttempts = 60;
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 2000));

    try {
      const res = await api(`/api/v1/documents/tasks/${taskId}`);

      if (res.status === "SUCCESS") {
        hide($("#upload-status"));
        const result = $("#upload-result");
        result.className = "success";
        result.innerHTML = `Document ingested successfully!<br>
          <small>Title: ${res.result?.title || "N/A"} | Chunks: ${res.result?.chunk_count || "N/A"} | Version: ${res.result?.version || "N/A"}</small>`;
        show(result);
        $("#upload-form").reset();
        $("#file-name").textContent = "No file selected";
        return;
      }

      if (res.status === "FAILURE") {
        throw new Error(res.error || "Processing failed");
      }

      // Still processing
      $("#upload-message").textContent = `Processing... (${res.status})`;

    } catch (err) {
      hide($("#upload-status"));
      const result = $("#upload-result");
      result.className = "error";
      result.textContent = err.message;
      show(result);
      return;
    }
  }

  hide($("#upload-status"));
  const result = $("#upload-result");
  result.className = "error";
  result.textContent = "Processing timed out. Check task status manually.";
  show(result);
}

// ── Init ─────────────────────────────────────────────────────────────────────

if (token) {
  enterApp();
}
