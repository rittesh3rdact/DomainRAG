const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const statusEl = document.getElementById("status");

let sessionId = localStorage.getItem("rag_session_id") || null;

function addBubble(role, text) {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    statusEl.textContent = `${data.indexed_chunks} chunk(s) indexed`;
  } catch {
    statusEl.textContent = "backend unreachable";
  }
}

async function sendMessage(message) {
  addBubble("user", message);
  const botBubble = addBubble("bot", "");
  const sourcesEl = document.createElement("div");
  sourcesEl.className = "sources";

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  if (!response.ok || !response.body) {
    const err = await response.json().catch(() => ({ detail: "Request failed" }));
    botBubble.textContent = err.detail || "Something went wrong.";
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop(); // last (possibly incomplete) chunk stays in buffer

    for (const raw of events) {
      if (!raw.startsWith("data: ")) continue;
      const event = JSON.parse(raw.slice(6));

      if (event.type === "token") {
        botBubble.textContent += event.text;
      } else if (event.type === "sources" && event.sources.length) {
        sourcesEl.textContent = `Sources: ${event.sources.join(", ")}`;
        botBubble.after(sourcesEl);
      } else if (event.type === "error") {
        botBubble.textContent = event.message;
      } else if (event.type === "done") {
        sessionId = event.session_id;
        localStorage.setItem("rag_session_id", sessionId);
      }
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = "";
  inputEl.disabled = true;
  formEl.querySelector("button").disabled = true;
  try {
    await sendMessage(message);
  } finally {
    inputEl.disabled = false;
    formEl.querySelector("button").disabled = false;
    inputEl.focus();
  }
});

refreshStatus();
