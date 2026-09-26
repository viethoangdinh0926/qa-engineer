const app = document.querySelector("#app");

const state = {
  runs: [],
  expandedId: null,
  notice: "",
  formError: "",
  sources: new Map(),
  plannerChatOpen: false,
  plannerChatMessages: [],
  currentPlan: null,
  currentRunId: null,
  busy: false,
};

function statusClass(value) {
  return String(value || "pending").replace(/[^a-z]/g, "");
}

function runLabel(run) {
  if (run.ready && run.report && run.report.verdict) {
    return run.report.verdict;
  }
  return run.status || "submitted";
}

function isSuccessOrFailure(run) {
  const verdict = run.report && run.report.verdict;
  return verdict === "pass" || verdict === "fail";
}

async function loadCapabilities() {
  const chips = document.querySelector("#chips");
  const response = await fetch("/v1/capabilities");
  const data = await response.json();
  chips.replaceChildren();
  for (const name of ["browser", "coding"]) {
    const chip = document.createElement("span");
    chip.className = data[name] ? "chip" : "chip off";
    const detail = data.detail && data.detail[name] ? ` — ${data.detail[name]}` : "";
    chip.textContent = data[name] ? `${name} available` : `${name} unavailable${detail}`;
    chips.appendChild(chip);
  }
}

function renderShell() {
  app.replaceChildren();
  
  // Create main layout with sidebar
  const mainLayout = document.createElement("div");
  mainLayout.className = "main-layout";
  
  // Sidebar for planner chat
  const sidebar = document.createElement("aside");
  sidebar.className = "sidebar";
  sidebar.id = "planner-sidebar";
  
  const sidebarHeader = document.createElement("div");
  sidebarHeader.className = "sidebar-header";
  const sidebarTitle = document.createElement("h3");
  sidebarTitle.textContent = "Planner Chat";
  const toggleButton = document.createElement("button");
  toggleButton.className = "toggle-sidebar";
  toggleButton.textContent = "×";
  toggleButton.addEventListener("click", () => {
    state.plannerChatOpen = false;
    sidebar.classList.remove("open");
  });
  sidebarHeader.append(sidebarTitle, toggleButton);
  
  const chatContainer = document.createElement("div");
  chatContainer.className = "chat-container";
  chatContainer.id = "chat-messages";
  
  const chatInput = document.createElement("div");
  chatInput.className = "chat-input";
  const chatTextarea = document.createElement("textarea");
  chatTextarea.id = "chat-message";
  chatTextarea.placeholder = "Ask the planner to modify the plan...";
  chatTextarea.rows = 3;
  const sendButton = document.createElement("button");
  sendButton.textContent = "Send";
  sendButton.addEventListener("click", () => sendPlannerMessage());
  chatInput.append(chatTextarea, sendButton);
  
  sidebar.append(sidebarHeader, chatContainer, chatInput);
  
  // Main content area
  const mainContent = document.createElement("main");
  mainContent.className = "main-content";
  
  const form = document.createElement("form");
  form.className = "card";
  const heading = document.createElement("h2");
  heading.textContent = "New test";
  const label = document.createElement("label");
  label.className = "field";
  label.htmlFor = "specification";
  label.textContent = "Testing request";
  const area = document.createElement("textarea");
  area.id = "specification";
  area.name = "specification";
  area.placeholder = "Describe what you want to test";
  const row = document.createElement("div");
  row.className = "row";
  const button = document.createElement("button");
  button.type = "submit";
  button.id = "submit-spec";
  button.textContent = "Create plan";
  row.append(button);
  const error = document.createElement("p");
  error.id = "form-error";
  error.className = "error";
  error.hidden = true;
  form.append(heading, label, area, row, error);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitSpecification(area, button);
  });

  const notice = document.createElement("p");
  notice.id = "notice";
  notice.className = "notice";
  notice.hidden = true;

  const section = document.createElement("section");
  section.className = "section";
  const runsHeading = document.createElement("h2");
  runsHeading.textContent = "Testing Runs";
  const list = document.createElement("div");
  list.id = "testing-runs";
  section.append(runsHeading, list);
  mainContent.append(form, notice, section);
  
  mainLayout.append(sidebar, mainContent);
  app.append(mainLayout);
  
  // Add toggle button for sidebar
  const sidebarToggle = document.createElement("button");
  sidebarToggle.className = "sidebar-toggle";
  sidebarToggle.textContent = "☰ Planner Chat";
  sidebarToggle.addEventListener("click", () => {
    state.plannerChatOpen = !state.plannerChatOpen;
    sidebar.classList.toggle("open", state.plannerChatOpen);
  });
  app.prepend(sidebarToggle);
}

function paintChrome() {
  const error = document.querySelector("#form-error");
  error.hidden = !state.formError;
  error.textContent = state.formError;
  const notice = document.querySelector("#notice");
  notice.hidden = !state.notice;
  notice.textContent = state.notice;
}

function setBusy(busy) {
  state.busy = busy;
  document.body.classList.toggle("busy", busy);
  const chatBox = document.querySelector("#chat-message");
  const send = document.querySelector(".chat-input button");
  if (chatBox) {
    chatBox.disabled = busy;
  }
  if (send) {
    send.disabled = busy;
  }
  renderChatMessages();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));
}

function executing(run) {
  return Boolean(run && !run.ready && (run.status === "running" || run.status === "working"));
}

function stepDetails(step, detailed) {
  const item = document.createElement("li");
  const tag = document.createElement("div");
  tag.className = "tag";
  if (step.interface === "GUI") {
    tag.textContent = `GUI ${step.gui_driver || ""}`.trim();
  } else if (step.interface === "CODING") {
    tag.textContent = "CODING";
  } else {
    tag.textContent = "CLI";
  }
  const body = document.createElement("div");
  const action = document.createElement("p");
  action.className = "action";
  action.textContent = step.phase_name
    ? `Phase ${step.phase || step.step}. ${step.phase_name}`
    : `${step.step}. ${step.action}`;
  body.append(action);
  if (step.depends_on && step.depends_on.length) {
    const depends = document.createElement("p");
    depends.className = "summary";
    depends.textContent = `After phase ${step.depends_on.join(", ")}`;
    body.append(depends);
  }
  if (step.operation_notes && step.operation_notes.length) {
    const operations = document.createElement("p");
    operations.className = "summary";
    operations.textContent = step.operation_notes.map((note, index) => `${index + 1}. ${note}`).join("\n");
    body.append(operations);
  }
  if (step.interface === "CLI" && step.script) {
    const script = document.createElement("pre");
    script.className = "evidence";
    script.textContent = step.script;
    body.append(script);
  }
  // Show coding operations for CODING steps
  if (step.interface === "CODING" && step.coding_operations && step.coding_operations.length) {
    const codingOps = document.createElement("p");
    codingOps.className = "summary";
    const opText = step.coding_operations.map((op, index) => {
      const opStr = `${index + 1}. ${op.action}`;
      if (op.file_path) {
        return `${opStr}: ${op.file_path}`;
      }
      return opStr;
    }).join("\n");
    codingOps.textContent = opText;
    body.append(codingOps);
  }
  const checks = step.verification_results || [];
  if (checks.length) {
    checks.forEach((check) => {
      const question = document.createElement("p");
      question.className = "summary";
      question.textContent = `Verification: ${check.question}`;
      body.append(question);
      if (detailed) {
        const judged = document.createElement("p");
        judged.className = "summary";
        const passed = check.passed;
        judged.textContent = `Verification result: ${passed === true ? "passed" : passed === false ? "failed" : "not judged"}`;
        body.append(judged);
        if (check.judgment) {
          const judgment = document.createElement("p");
          judgment.className = "summary";
          judgment.textContent = `Judgment: ${check.judgment}`;
          body.append(judgment);
        }
      }
    });
  } else if (detailed) {
    const assertion = document.createElement("p");
    assertion.className = "summary";
    assertion.textContent = `Assertion: ${step.assertion || "none"}`;
    const judged = document.createElement("p");
    judged.className = "summary";
    const passed = step.assertion_passed;
    judged.textContent = `Assertion result: ${passed === true ? "passed" : passed === false ? "failed" : "not judged"}`;
    body.append(assertion, judged);
    if (step.judgment) {
      const judgment = document.createElement("p");
      judgment.className = "summary";
      judgment.textContent = `Judgment: ${step.judgment}`;
      body.append(judgment);
    }
  }
  if (detailed && step.summary) {
    const summary = document.createElement("p");
    summary.className = "summary";
    summary.textContent = step.summary;
    body.append(summary);
  }
  if (detailed) {
    const evidence = step.evidence || {};
    const keys = Object.keys(evidence).filter((key) => key !== "page_source" && evidence[key]);
    if (keys.length) {
      const block = document.createElement("pre");
      block.className = "evidence";
      block.textContent = keys.map((key) => `${key}: ${evidence[key]}`).join("\n");
      body.append(block);
    }
  } else if (!checks.length && step.summary) {
    const summary = document.createElement("p");
    summary.className = "summary";
    summary.textContent = step.summary;
    body.append(summary);
  }
  const status = document.createElement("div");
  status.className = `status ${statusClass(step.status)}`;
  status.textContent = step.status;
  item.append(tag, body, status);
  return item;
}

function paintRuns() {
  const list = document.querySelector("#testing-runs");
  list.replaceChildren();
  if (!state.runs.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No runs yet.";
    list.append(empty);
    return;
  }
  const items = document.createElement("div");
  items.className = "run-list";
  for (const run of state.runs) {
    const block = document.createElement("article");
    block.className = "run-block";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-toggle";
    button.setAttribute("aria-expanded", String(state.expandedId === run.id));
    const id = document.createElement("span");
    id.className = "run-id";
    id.textContent = run.id;
    const status = document.createElement("span");
    const label = runLabel(run);
    status.className = `status ${statusClass(label)}`;
    status.textContent = label;
    button.append(id, status);
    button.addEventListener("click", () => toggleRun(run.id));
    block.append(button);
    if (state.expandedId === run.id) {
      block.append(runPanel(run));
    }
    items.append(block);
  }
  list.append(items);
}

function runPanel(run) {
  const panel = document.createElement("div");
  panel.className = "run-panel";
  const banner = document.createElement("div");
  const label = runLabel(run);
  banner.className = `banner ${statusClass(label)}`;
  const verdict = document.createElement("strong");
  verdict.textContent = label;
  banner.append(verdict);
  panel.append(banner);
  const specText = (run.report && run.report.specification) || run.specification;
  if (specText) {
    const spec = document.createElement("section");
    spec.className = "spec";
    const heading = document.createElement("h3");
    heading.textContent = "Specification";
    const body = document.createElement("pre");
    body.textContent = specText;
    spec.append(heading, body);
    panel.append(spec);
  }
  if (run.report && run.report.reason) {
    const reason = document.createElement("p");
    reason.className = "reason";
    reason.textContent = run.report.reason;
    panel.append(reason);
  }
  if (run.report && run.report.reason_code === "missing_capability") {
    const missing = document.createElement("p");
    missing.className = "summary";
    missing.textContent = `Missing: ${(run.report.missing || []).join(", ")}`;
    panel.append(missing);
  }
  const detailed = isSuccessOrFailure(run);
  const steps = document.createElement("ol");
  steps.className = "steps";
  for (const step of run.steps || []) {
    steps.append(stepDetails(step, detailed));
  }
  if ((run.steps || []).length) {
    panel.append(steps);
  } else if (!run.ready) {
    const waiting = document.createElement("p");
    waiting.className = "summary";
    waiting.textContent = "Waiting for the first step.";
    panel.append(waiting);
  }
  
  
  return panel;
}

function upsertRun(snapshot) {
  const index = state.runs.findIndex((run) => run.id === snapshot.id);
  if (index === -1) {
    state.runs.unshift(snapshot);
  } else {
    state.runs[index] = snapshot;
  }
  paintRuns();
  if (snapshot.id !== state.currentRunId) {
    return;
  }
  if (snapshot.plan) {
    state.currentPlan = snapshot.plan;
    renderPlan();
    return;
  }
  if (state.currentPlan) {
    loadPlan(snapshot.id);
  }
}

function toggleRun(id) {
  if (state.busy) {
    return;
  }
  if (state.expandedId === id) {
    state.expandedId = null;
    window.history.pushState({}, "", "/");
    paintRuns();
    return;
  }
  openRun(id);
}

async function openRun(id) {
  state.expandedId = id;
  state.currentRunId = id;
  state.currentPlan = null;
  state.plannerChatMessages = [];
  state.plannerChatOpen = true;
  window.history.pushState({}, "", `/runs/${id}`);
  const sidebar = document.querySelector("#planner-sidebar");
  if (sidebar) {
    sidebar.classList.add("open");
  }
  const response = await fetch(`/v1/runs/${id}`);
  if (response.ok) {
    upsertRun(await response.json());
  }
  await loadPlan(id);
  await loadChatHistory(id);
  const run = state.runs.find((item) => item.id === id);
  if (executing(run)) {
    watchRun(id);
  }
  paintRuns();
}

function watchRun(id) {
  if (state.sources.has(id)) {
    return;
  }
  const source = new EventSource(`/v1/runs/${id}/events`);
  state.sources.set(id, source);
  let finished = false;
  source.onmessage = (event) => {
    const snapshot = JSON.parse(event.data);
    upsertRun(snapshot);

    if (snapshot.ready) {
      finished = true;
      source.close();
      state.sources.delete(id);
    }
  };
  source.onerror = () => {
    source.close();
    state.sources.delete(id);
    if (!finished) {
      window.setTimeout(() => watchRun(id), 1000);
    }
  };
}

async function submitSpecification(area, button) {
  if (state.busy) {
    return;
  }
  state.formError = "";
  state.notice = "";
  paintChrome();
  const sidebar = document.querySelector("#planner-sidebar");
  if (sidebar) {
    sidebar.classList.add("open");
    state.plannerChatOpen = true;
  }
  setBusy(true);
  button.disabled = true;
  try {
    const response = await fetch("/v1/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ specification: area.value, wait_for_approval: true }),
    });
    const body = await response.json();
    if (!response.ok) {
      state.formError = body.error || "The specification was rejected.";
      paintChrome();
      return;
    }
    area.value = "";
    state.notice = `Plan ready for ${body.id}`;
    paintChrome();
    await openRun(body.id);
  } finally {
    button.disabled = false;
    setBusy(false);
  }
}

async function loadPlan(runId) {
  try {
    const response = await fetch(`/v1/runs/${runId}/plan`);
    state.currentPlan = response.ok ? await response.json() : null;
    renderPlan();
  } catch (error) {
    console.error("Failed to load plan:", error);
  }
}

async function loadChatHistory(runId) {
  try {
    const response = await fetch(`/v1/runs/${runId}/planner/chat/history`);
    if (response.ok) {
      const data = await response.json();
      state.plannerChatMessages = data.messages || [];
      renderChatMessages();
    }
  } catch (error) {
    console.error("Failed to load chat history:", error);
  }
}

async function sendPlannerMessage() {
  const textarea = document.getElementById("chat-message");
  const message = textarea.value.trim();
  if (!message || !state.currentRunId || state.busy) return;

  textarea.value = "";
  state.plannerChatMessages.push({ role: "user", content: message });
  setBusy(true);

  try {
    const response = await fetch(`/v1/runs/${state.currentRunId}/planner/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    if (response.ok) {
      state.plannerChatMessages = data.messages || [];
      await loadPlan(state.currentRunId);
    } else {
      state.notice = data.error || "The agent could not answer.";
      paintChrome();
      await loadChatHistory(state.currentRunId);
    }
  } catch (error) {
    console.error("Failed to send message:", error);
    state.notice = "The agent could not answer.";
    paintChrome();
  } finally {
    setBusy(false);
  }
}

function renderChatMessages() {
  const container = document.getElementById("chat-messages");
  if (!container) return;
  
  container.replaceChildren();
  
  state.plannerChatMessages.forEach((msg) => {
    const messageDiv = document.createElement("div");
    messageDiv.className = `chat-message ${msg.role}`;
    const roleLabel = document.createElement("span");
    roleLabel.className = "chat-role";
    roleLabel.textContent = msg.role === "user" ? "You:" : "Planner:";
    const content = document.createElement("p");
    content.textContent = msg.content;
    messageDiv.append(roleLabel, content);
    container.appendChild(messageDiv);
  });
  
  if (state.busy) {
    const pending = document.createElement("div");
    pending.className = "chat-message processing";
    pending.textContent = "processing…";
    container.appendChild(pending);
  }

  container.scrollTop = container.scrollHeight;
}

function renderPlan() {
  if (!state.currentPlan) {
    const existing = document.getElementById("plan-section");
    if (existing) {
      existing.replaceChildren();
    }
    return;
  }
  
  // Create plan display section
  let planSection = document.getElementById("plan-section");
  if (!planSection) {
    planSection = document.createElement("section");
    planSection.className = "section";
    planSection.id = "plan-section";
    const mainContent = document.querySelector(".main-content");
    if (mainContent) {
      mainContent.appendChild(planSection);
    }
  }
  
  planSection.replaceChildren();
  
  const heading = document.createElement("h2");
  heading.textContent = "Test Plan";
  
  const statusBadge = document.createElement("span");
  statusBadge.className = `plan-status ${state.currentPlan.status}`;
  statusBadge.textContent = state.currentPlan.status;
  heading.append(statusBadge);
  
  const planContent = document.createElement("div");
  planContent.className = "plan-content";
  
  // Render phases - use direct phases field (simplified storage)
  const phases = state.currentPlan.phases || [];
  phases.forEach((phase) => {
    const phaseDiv = document.createElement("div");
    phaseDiv.className = "plan-phase";
    phaseDiv.innerHTML = `
      <h4>Phase ${phase.phase}: ${phase.name}</h4>
      <p><strong>Interface:</strong> ${phase.interface}</p>
      <p><strong>Depends on:</strong> ${(phase.depends_on || []).join(", ") || "None"}</p>
      <p><strong>Operations:</strong></p>
      <ul>
        ${(phase.operation_notes || []).map((note) => `<li>${note}</li>`).join("")}
        ${(phase.coding_operations || []).map((op) => `<li>${op.action}${op.file_path ? `: ${op.file_path}` : ""}</li>`).join("")}
      </ul>
      ${phase.interface === "CLI" && phase.script ? `<p><strong>Script:</strong></p><pre>${escapeHtml(phase.script)}</pre>` : ""}
      <p><strong>Verifications:</strong></p>
      <ul>
        ${(phase.verifications || []).map(v => `<li>${v}</li>`).join("")}
      </ul>
    `;
    planContent.appendChild(phaseDiv);
  });
  
  // Add action buttons
  const actionsDiv = document.createElement("div");
  actionsDiv.className = "plan-actions";
  
  const run = state.runs.find((item) => item.id === state.currentRunId);
  if (executing(run)) {
    const cancelButton = document.createElement("button");
    cancelButton.textContent = "Cancel execution";
    cancelButton.className = "reject-btn";
    cancelButton.addEventListener("click", () => cancelExecution());
    actionsDiv.appendChild(cancelButton);
  } else if (state.currentPlan.status !== "rejected" && (state.currentPlan.phases || []).length) {
    const runButton = document.createElement("button");
    runButton.className = "approve-btn";
    if (state.currentPlan.status === "approved") {
      runButton.textContent = "Run plan";
      runButton.addEventListener("click", () => rerunPlan());
    } else {
      runButton.textContent = "Approve and run";
      runButton.addEventListener("click", () => approvePlan());
    }
    actionsDiv.appendChild(runButton);
  }
  
  const downloadButton = document.createElement("button");
  downloadButton.textContent = "Download Plan (JSON)";
  downloadButton.className = "download-btn";
  downloadButton.addEventListener("click", () => downloadPlan("json"));
  
  actionsDiv.appendChild(downloadButton);
  
  planSection.append(heading, planContent, actionsDiv);
}

async function approvePlan() {
  if (!state.currentRunId || state.busy) return;
  setBusy(true);
  try {
    const response = await fetch(`/v1/runs/${state.currentRunId}/plan/approve`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ user: "user" }),
    });
    const body = await response.json();
    if (!response.ok) {
      state.notice = body.error || "Could not start the run.";
      paintChrome();
      return;
    }
    state.currentPlan = body;
    await refreshSelectedRun();
    watchRun(state.currentRunId);
    renderPlan();
  } catch (error) {
    console.error("Failed to approve plan:", error);
  } finally {
    setBusy(false);
  }
}

async function refreshSelectedRun() {
  if (!state.currentRunId) return;
  const response = await fetch(`/v1/runs/${state.currentRunId}`);
  if (response.ok) {
    upsertRun(await response.json());
  }
}

async function cancelExecution() {
  if (!state.currentRunId || state.busy) return;
  const response = await fetch(`/v1/runs/${state.currentRunId}:cancel`, { method: "POST" });
  if (!response.ok) {
    const body = await response.json();
    state.notice = body.error || "Could not cancel the run.";
    paintChrome();
    return;
  }
  upsertRun(await response.json());
  watchRun(state.currentRunId);
  renderPlan();
}

async function rerunPlan() {
  if (!state.currentRunId || state.busy) return;
  setBusy(true);
  try {
    const response = await fetch(`/v1/runs/${state.currentRunId}:start`, { method: "POST" });
    const body = await response.json();
    if (!response.ok) {
      state.notice = body.error || "Could not start the run.";
      paintChrome();
      return;
    }
    upsertRun(body);
    watchRun(state.currentRunId);
    renderPlan();
  } catch (error) {
    console.error("Failed to re-run plan:", error);
  } finally {
    setBusy(false);
  }
}

async function rejectPlan() {
  if (!state.currentRunId) return;
  
  const reason = prompt("Reason for rejection:");
  if (!reason) return;
  
  try {
    const response = await fetch(`/v1/runs/${state.currentRunId}/plan/reject`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ reason }),
    });
    if (response.ok) {
      state.currentPlan = await response.json();
      renderPlan();
    }
  } catch (error) {
    console.error("Failed to reject plan:", error);
  }
}

async function downloadPlan(format) {
  if (!state.currentRunId) return;
  
  try {
    const response = await fetch(`/v1/runs/${state.currentRunId}/plan/download?format=${format}`);
    if (response.ok) {
      const data = await response.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `plan-${state.currentRunId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  } catch (error) {
    console.error("Failed to download plan:", error);
  }
}

async function boot() {
  await loadCapabilities();
  renderShell();
  const response = await fetch("/v1/runs");
  const body = await response.json();
  state.runs = body.runs || [];
  const match = window.location.pathname.match(/^\/runs\/([^/]+)$/);
  if (match) {
    await openRun(decodeURIComponent(match[1]));
  } else {
    paintRuns();
  }
  for (const run of state.runs) {
    if (executing(run)) {
      watchRun(run.id);
    }
  }
  window.addEventListener("popstate", () => {
    const next = window.location.pathname.match(/^\/runs\/([^/]+)$/);
    if (next) {
      openRun(decodeURIComponent(next[1]));
    } else {
      state.expandedId = null;
      paintRuns();
    }
  });
}

boot();
