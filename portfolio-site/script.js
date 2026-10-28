const API_BASE = "/api";

let isAuthenticatedGlobal = false;
let accessToken = null;

// ── On page load ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  // Restore session from localStorage
  const stored = localStorage.getItem("cerince_token");
  if (stored) {
    accessToken = stored;
    isAuthenticatedGlobal = true;
  }
  updateUI(isAuthenticatedGlobal);

  // Login form submit
  document.getElementById("loginForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = document.getElementById("loginUsername").value.trim();
    const password = document.getElementById("loginPassword").value;
    const errorEl = document.getElementById("loginError");
    errorEl.textContent = "";

    try {
      console.log(`[LOGIN] Attempting login for username="${username}" → POST ${API_BASE}/login`);
      const res = await fetch(`${API_BASE}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      console.log(`[LOGIN] Response: HTTP ${res.status} ${res.statusText}`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        console.warn("[LOGIN] Failed:", data);
        throw new Error(data.detail || "Login failed");
      }
      const data = await res.json();
      accessToken = data.access_token;
      localStorage.setItem("cerince_token", accessToken);
      isAuthenticatedGlobal = true;
      console.log("[LOGIN] Success — token stored in localStorage");
      closeLoginModal();
      updateUI(true);
    } catch (err) {
      console.error("[LOGIN] Error:", err);
      errorEl.textContent = err.message;
    }
  });

  // Login / Logout buttons
  document.getElementById("loginButton")?.addEventListener("click", openLoginModal);
  document.getElementById("logoutButton")?.addEventListener("click", () => {
    accessToken = null;
    isAuthenticatedGlobal = false;
    localStorage.removeItem("cerince_token");
    updateUI(false);
  });

  // Close modal on backdrop click
  document.getElementById("loginModal")?.addEventListener("click", (e) => {
    if (e.target === document.getElementById("loginModal")) closeLoginModal();
  });

  // Fetch and display projects
  fetch("projects.json")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return response.json();
    })
    .then((projects) => {
      const projectList = document.getElementById("projectList");
      if (!projectList) return;
      projectList.innerHTML = "";

      projects.forEach((project) => {
        const card = document.createElement("article");
        card.className = "card";

        let linkText = "View Project";
        if (project.link) {
          if (project.link.includes("github.com")) linkText = "View on GitHub";
          else if (project.link.includes("behance.net")) linkText = "View on Behance";
          else if (project.link.includes("figma.com")) linkText = "View on Figma";
        }

        card.innerHTML = `
          <div class="card-body">
            <h3>${project.title}</h3>
            <p>${project.description}</p>
            <div class="tech-tags">
              ${project.tech.map((t) => `<span class="tag">${t}</span>`).join("")}
            </div>
          </div>
          ${
            project.link
              ? `<footer class="card-footer">
                   <a href="${project.link}" target="_blank" role="button" class="contrast">${linkText}</a>
                 </footer>`
              : ""
          }
        `;
        projectList.appendChild(card);
      });
    })
    .catch((error) => {
      console.error("Error loading projects:", error);
      const projectList = document.getElementById("projectList");
      if (projectList) {
        projectList.innerHTML = '<p style="color: red;">Error loading projects. Check console for details.</p>';
      }
    });

  // Chat toggle
  const chatToggleButton = document.getElementById("chatToggle");
  const chatBox = document.getElementById("chatBox");
  const chatInput = document.getElementById("chatInput");
  const closeChatButton = document.getElementById("closeChat");

  if (chatToggleButton && chatBox && chatInput) {
    chatToggleButton.addEventListener("click", () => {
      chatBox.classList.toggle("hidden");
      if (!chatBox.classList.contains("hidden")) {
        chatInput.focus();
        if (!hasShownWelcomeMessage) {
          const intro = isAuthenticatedGlobal
            ? "Hi! I'm Cerince, your AI assistant. How can I help you today?"
            : "Hi! Login to start chatting with the AI.";
          addMessage(intro, false);
          hasShownWelcomeMessage = true;
        }
      }
    });
  }

  closeChatButton?.addEventListener("click", () => {
    chatBox?.classList.add("hidden");
  });

  const chatForm = document.querySelector("#chatBox form");
  chatForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage();
  });
});

// ── Modal helpers ──────────────────────────────────────────────────────────
function openLoginModal() {
  const modal = document.getElementById("loginModal");
  if (modal) {
    modal.style.display = "flex";
    document.getElementById("loginUsername")?.focus();
  }
}

function closeLoginModal() {
  const modal = document.getElementById("loginModal");
  if (modal) modal.style.display = "none";
}

// ── UI Helpers ─────────────────────────────────────────────────────────────
function updateUI(isAuthenticated) {
  document.getElementById("loginButton").style.display = isAuthenticated ? "none" : "block";
  document.getElementById("logoutButton").style.display = isAuthenticated ? "block" : "none";

  const toggle = document.getElementById("chatToggle");
  if (toggle) toggle.style.display = "flex";

  const input = document.getElementById("chatInput");
  const formButton = document.querySelector('#chatBox button[type="submit"]');
  if (input) {
    input.disabled = !isAuthenticated;
    input.placeholder = isAuthenticated ? "Ask me anything..." : "Login required to chat";
  }
  if (formButton) {
    formButton.disabled = !isAuthenticated;
    formButton.title = isAuthenticated ? "Send" : "Login to enable chat";
  }
}

function addMessage(content, isUser) {
  const messagesContainer = document.getElementById("chatMessages");
  if (!messagesContainer) return;

  const messageWrapper = document.createElement("div");
  messageWrapper.className = `message-wrapper ${isUser ? "user" : "bot"}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = isUser
    ? `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>`
    : `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="8" width="18" height="12" rx="2"/><path d="M8 8V6a4 4 0 0 1 8 0v2"/><circle cx="12" cy="14" r="2"/></svg>`;

  const messageContent = document.createElement("div");
  messageContent.className = "message";
  messageContent.textContent = content;

  const timestamp = document.createElement("div");
  timestamp.className = "timestamp";
  timestamp.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  messageContent.appendChild(timestamp);
  messageWrapper.appendChild(avatar);
  messageWrapper.appendChild(messageContent);
  messagesContainer.appendChild(messageWrapper);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

let typingIndicatorElement = null;
let isProcessingMessage = false;
let hasShownWelcomeMessage = false;

function showTypingIndicator() {
  const messagesContainer = document.getElementById("chatMessages");
  if (!messagesContainer || typingIndicatorElement) return;

  const wrapper = document.createElement("div");
  wrapper.className = "message-wrapper bot";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="8" width="18" height="12" rx="2"/><path d="M8 8V6a4 4 0 0 1 8 0v2"/><circle cx="12" cy="14" r="2"/></svg>`;

  const indicator = document.createElement("div");
  indicator.className = "message typing-indicator";
  indicator.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;

  wrapper.appendChild(avatar);
  wrapper.appendChild(indicator);
  messagesContainer.appendChild(wrapper);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  typingIndicatorElement = wrapper;
}

function hideTypingIndicator() {
  if (typingIndicatorElement && typingIndicatorElement.parentNode) {
    typingIndicatorElement.parentNode.removeChild(typingIndicatorElement);
  }
  typingIndicatorElement = null;
}

async function sendMessage() {
  if (isProcessingMessage) return;

  const input = document.getElementById("chatInput");
  const spinner = document.getElementById("chatSpinner");
  const query = input.value.trim();
  if (!query) return;

  if (!isAuthenticatedGlobal) {
    openLoginModal();
    return;
  }

  isProcessingMessage = true;
  input.value = "";
  input.disabled = true;

  addMessage(query, true);
  if (spinner) spinner.style.display = "inline-block";
  showTypingIndicator();

  try {
    console.log(`[CHAT] Sending query → POST ${API_BASE}/chat`, { query, n_results: 5 });
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ query, n_results: 5 }),
    });

    console.log(`[CHAT] Response: HTTP ${response.status} ${response.statusText}`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      console.warn("[CHAT] Non-OK response body:", errorData);
      if (response.status === 401) {
        accessToken = null;
        isAuthenticatedGlobal = false;
        localStorage.removeItem("cerince_token");
        updateUI(false);
        throw new Error("Session expired. Please log in again.");
      }
      throw new Error(errorData?.detail || `HTTP ${response.status}`);
    }

    // --- Create the bot message bubble before the first token arrives ---
    const messagesContainer = document.getElementById("chatMessages");
    const messageWrapper = document.createElement("div");
    messageWrapper.className = "message-wrapper bot";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="8" width="18" height="12" rx="2"/><path d="M8 8V6a4 4 0 0 1 8 0v2"/><circle cx="12" cy="14" r="2"/></svg>`;

    const messageContent = document.createElement("div");
    messageContent.className = "message";

    const textNode = document.createTextNode("");
    messageContent.appendChild(textNode);

    const timestamp = document.createElement("div");
    timestamp.className = "timestamp";
    timestamp.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    messageContent.appendChild(timestamp);

    messageWrapper.appendChild(avatar);
    messageWrapper.appendChild(messageContent);

    // --- Consume the SSE stream ---
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let firstToken = true;
    let tokenCount = 0;

    console.log("[CHAT] SSE stream open — reading tokens ...");

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log(`[CHAT] SSE stream closed by server (tokenCount=${tokenCount})`);
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      // SSE lines: "data: {...}\n\n"
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // keep any incomplete tail

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;

        let payload;
        try {
          payload = JSON.parse(trimmed.slice(5).trim());
        } catch {
          console.warn("[CHAT] Could not parse SSE line:", trimmed);
          continue;
        }

        if (payload.error) {
          console.error("[CHAT] Server sent error payload:", payload.error);
          // Replace any streamed content with the error message
          textNode.textContent = `Error: ${payload.error}`;
          break;
        }

        if (payload.done) {
          console.log(`[CHAT] SSE done signal received (tokenCount=${tokenCount})`);
          break;
        }

        if (payload.token) {
          tokenCount++;
          if (firstToken) {
            // Remove typing indicator and mount the bubble on first real token
            console.log("[CHAT] First token received — displaying bot bubble");
            hideTypingIndicator();
            messagesContainer.appendChild(messageWrapper);
            firstToken = false;
          }
          if (tokenCount % 10 === 0) {
            console.debug(`[CHAT] Streaming... tokenCount=${tokenCount}`);
          }
          textNode.textContent += payload.token;
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
      }
    }

    // If no tokens ever arrived, show a fallback message
    if (firstToken) {
      console.warn("[CHAT] No tokens received from SSE stream — showing fallback");
      hideTypingIndicator();
      addMessage("No response received.", false);
    }

  } catch (error) {
    const friendly = error && typeof error.message === "string" ? error.message : "Unexpected error";
    console.error("[CHAT] Fetch/stream error:", error);
    addMessage(`Error: ${friendly}. Please try again later.`, false);
    console.error("Fetch Error:", error);
  } finally {
    if (spinner) spinner.style.display = "none";
    hideTypingIndicator(); // safe no-op if already removed
    input.disabled = false;
    input.focus();
    isProcessingMessage = false;
  }
}
