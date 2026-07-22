(() => {
  "use strict";

  const CONFIG_KEY = "odr.config.v3";
  const DATA_KEY = "odr.workspace.v3";
  const PHASES = ["scope", "plan", "research", "report"];
  const PHASE_LABELS = {
    scope: ["理解问题", "判断是否需要澄清"],
    plan: ["制定方案", "等待确认研究大纲"],
    research: ["检索与分析", "并行搜索并交叉验证"],
    report: ["生成报告", "整理结论与引用"],
  };
  const DEFAULT_CONFIG = {
    search_apis: ["tavily"],
    max_search_calls: 6,
    max_concurrent_research_units: 5,
    max_researcher_iterations: 6,
    max_react_tool_calls: 10,
    allow_clarification: false,
    model_provider: "deepseek",
    model_name: "deepseek:deepseek-chat",
    same_model: true,
    research_model: "deepseek:deepseek-chat",
    summarization_model: "deepseek:deepseek-chat",
    compression_model: "deepseek:deepseek-chat",
    final_report_model: "deepseek:deepseek-chat",
    ollama_base_url: "http://host.docker.internal:11434",
    knowledge_bases: [],
    mcp_enabled: false,
    mcp_url: "",
    mcp_tools: [],
    mcp_auth_required: false,
    mcp_prompt: "",
    generate_report: true,
    report_language: "zh-CN",
    report_length: "standard",
    require_citations: true,
    api_base: "/api",
    theme: "light",
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    sidebar: $("sidebar"), mobileOverlay: $("mobileOverlay"), threadList: $("threadList"),
    welcome: $("welcome"), messageStack: $("messageStack"), conversation: $("conversation"),
    input: $("researchInput"), send: $("sendButton"), pageTitle: $("pageTitle"),
    pageSubtitle: $("pageSubtitle"), runBadge: $("runBadge"), showReport: $("showReport"),
    artifactPanel: $("artifactPanel"), artifactContent: $("artifactContent"),
    settings: $("settingsDrawer"), drawerOverlay: $("drawerOverlay"), toast: $("toast"),
    taskPanel: $("taskPanel"), taskPanelBody: $("taskPanelBody"), elapsedTime: $("elapsedTime"),
  };

  let config = loadJson(CONFIG_KEY, DEFAULT_CONFIG);
  config = { ...DEFAULT_CONFIG, ...config };
  let persisted = loadJson(DATA_KEY, {});
  const state = {
    threads: Array.isArray(persisted.threads) ? persisted.threads : [],
    currentThreadId: persisted.currentThreadId || null,
    messages: persisted.messages || {},
    runs: persisted.runs || {},
    reports: persisted.reports || {},
    streaming: false,
    controller: null,
    elapsedTimer: null,
  };

  function loadJson(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) || structuredClone(fallback); }
    catch { return structuredClone(fallback); }
  }

  function persist() {
    localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
    localStorage.setItem(DATA_KEY, JSON.stringify({
      threads: state.threads.slice(0, 30), currentThreadId: state.currentThreadId,
      messages: state.messages, runs: state.runs, reports: state.reports,
    }));
  }

  function escapeHtml(value = "") {
    return String(value).replace(/[&<>'"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[c]);
  }

  function md(value = "") {
    if (window.marked) return window.marked.parse(String(value), { breaks: true, gfm: true });
    return `<p>${escapeHtml(value).replace(/\n/g, "<br>")}</p>`;
  }

  function timeLabel(ts = Date.now()) {
    const d = new Date(ts);
    return d.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" }) + " " + d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  }

  function toast(text) {
    els.toast.textContent = text;
    els.toast.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => els.toast.classList.remove("show"), 2200);
  }

  function api(path) {
    return `${String(config.api_base || "/api").replace(/\/$/, "")}${path}`;
  }

  function currentMessages() {
    if (!state.currentThreadId) return [];
    return state.messages[state.currentThreadId] || (state.messages[state.currentThreadId] = []);
  }

  function currentRun() {
    return state.currentThreadId ? state.runs[state.currentThreadId] : null;
  }

  function addMessage(message) {
    currentMessages().push({ id: crypto.randomUUID(), createdAt: Date.now(), ...message });
    persist();
    render();
  }

  function titleFromQuery(query) {
    const compact = query.replace(/\s+/g, " ").trim();
    return compact.length > 25 ? `${compact.slice(0, 25)}…` : compact;
  }

  async function createThread(query) {
    const response = await fetch(api("/threads"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, configurable: buildRuntimeConfig() }),
    });
    if (!response.ok) throw new Error(await response.text() || "创建研究失败");
    const data = await response.json();
    const thread = { id: data.thread_id || data.id, title: data.title || titleFromQuery(query), createdAt: Date.now(), updatedAt: Date.now() };
    state.threads.unshift(thread);
    state.currentThreadId = thread.id;
    state.messages[thread.id] = [];
    state.reports[thread.id] = "";
    persist();
    return thread.id;
  }

  function buildRuntimeConfig() {
    const selected = config.search_apis.length ? config.search_apis : ["duckduckgo"];
    const model = config.model_name.trim() || "deepseek:deepseek-chat";
    const same = config.same_model;
    return {
      search_api: selected[0], search_apis: selected,
      max_search_calls: Number(config.max_search_calls),
      max_concurrent_research_units: Number(config.max_concurrent_research_units),
      max_researcher_iterations: Number(config.max_search_calls),
      max_react_tool_calls: Math.max(4, Number(config.max_search_calls) * 2),
      allow_clarification: false,
      research_model: same ? model : config.research_model,
      summarization_model: same ? model : config.summarization_model,
      compression_model: same ? model : config.compression_model,
      final_report_model: same ? model : config.final_report_model,
      ollama_base_url: config.ollama_base_url,
      knowledge_base_ids: config.knowledge_bases.map((item) => item.id),
      mcp_config: config.mcp_enabled && config.mcp_url ? {
        url: config.mcp_url, tools: config.mcp_tools, auth_required: config.mcp_auth_required,
      } : null,
      mcp_prompt: config.mcp_prompt,
      generate_report: config.generate_report,
      report_language: config.report_language,
      report_length: config.report_length,
      require_citations: config.require_citations,
    };
  }

  function newRun() {
    return {
      status: "running", title: "正在理解研究问题", subtitle: "AI 正在判断问题是否需要澄清",
      startedAt: Date.now(), endedAt: null, searchCount: 0, budget: Number(config.max_search_calls),
      phases: { scope: "running", plan: "pending", research: "pending", report: "pending" },
      activities: [], pending: null,
    };
  }

  async function startResearch() {
    const query = els.input.value.trim();
    if (!query || state.streaming) return;
    els.input.value = "";
    resizeInput();
    try {
      if (!state.currentThreadId || currentMessages().length) await createThread(query);
      const thread = state.threads.find((item) => item.id === state.currentThreadId);
      if (thread) thread.updatedAt = Date.now();
      addMessage({ role: "user", kind: "text", content: query });
      state.runs[state.currentThreadId] = newRun();
      startElapsedTimer();
      persist(); render();
      await streamRequest(api(`/threads/${encodeURIComponent(state.currentThreadId)}/runs/stream`), {
        messages: [{ role: "user", content: query }], configurable: buildRuntimeConfig(),
      });
    } catch (error) { handleError(error); }
  }

  async function resumeResearch(decision, messageId) {
    if (state.streaming || !state.currentThreadId) return;
    const message = currentMessages().find((item) => item.id === messageId);
    if (message) { message.resolved = true; message.resolution = decision.action; }
    const run = currentRun();
    if (run) { run.status = "running"; run.pending = null; }
    persist(); render();
    try {
      await streamRequest(api(`/threads/${encodeURIComponent(state.currentThreadId)}/runs/resume`), { resume: decision });
    } catch (error) { handleError(error); }
  }

  async function streamRequest(url, body) {
    state.streaming = true;
    state.controller = new AbortController();
    renderHeader();
    const response = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body), signal: state.controller.signal,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `请求失败 (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        consumeEvent(block);
      }
    }
    if (buffer.trim()) consumeEvent(buffer);
    state.streaming = false; state.controller = null;
    const run = currentRun();
    if (run && !run.endedAt && ["completed", "failed", "stopped"].includes(run.status)) run.endedAt = Date.now();
    stopElapsedTimer();
    persist(); render();
  }

  async function stopResearch() {
    const run = currentRun();
    if (!run || !["running", "waiting"].includes(run.status)) return;
    const threadId = state.currentThreadId;
    if (state.controller) state.controller.abort();
    state.streaming = false;
    state.controller = null;
    run.status = "stopped";
    run.endedAt = Date.now();
    run.title = "研究已停止";
    run.subtitle = "已保留停止前收集的资料与进度";
    const activePhase = PHASES.find((phase) => run.phases[phase] === "running");
    if (activePhase) run.phases[activePhase] = "stopped";
    stopElapsedTimer();
    persist(); render(); toast("已停止当前研究");
    try { await fetch(api(`/threads/${encodeURIComponent(threadId)}/runs/stop`), { method: "POST" }); } catch { /* local abort already completed */ }
  }

  function elapsedSeconds(run = currentRun()) {
    if (!run?.startedAt) return 0;
    return Math.max(0, Math.floor(((run.endedAt || Date.now()) - run.startedAt) / 1000));
  }

  function formatElapsed(seconds) {
    if (seconds < 60) return `${seconds} 秒`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes} 分 ${seconds % 60} 秒`;
  }

  function startElapsedTimer() {
    stopElapsedTimer();
    state.elapsedTimer = setInterval(() => {
      if (els.elapsedTime) els.elapsedTime.textContent = formatElapsed(elapsedSeconds());
    }, 1000);
  }

  function stopElapsedTimer() {
    if (state.elapsedTimer) clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
  }

  function consumeEvent(block) {
    let event = "message";
    const dataLines = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    let payload;
    try { payload = JSON.parse(dataLines.join("\n")); }
    catch { payload = { message: dataLines.join("\n") }; }
    handleEvent(event, payload);
  }

  function handleEvent(event, payload) {
    payload = payload || {};
    const run = currentRun() || newRun();
    state.runs[state.currentThreadId] = run;
    if (event === "run") {
      run.status = payload.status || "running";
    } else if (event === "progress") {
      const phase = payload.phase;
      if (phase && run.phases[phase] !== undefined) {
        for (const key of PHASES) {
          if (PHASES.indexOf(key) < PHASES.indexOf(phase) && run.phases[key] !== "failed") run.phases[key] = "completed";
        }
        run.phases[phase] = payload.status || "running";
      }
      run.title = payload.title || run.title;
      run.subtitle = payload.detail || payload.subtitle || PHASE_LABELS[phase]?.[1] || run.subtitle;
    } else if (event === "activity") {
      const activity = { id: payload.id || crypto.randomUUID(), at: Date.now(), status: payload.status || "running", ...payload };
      const existing = run.activities.findIndex((item) => item.id === activity.id);
      if (existing >= 0) run.activities[existing] = { ...run.activities[existing], ...activity };
      else run.activities.push(activity);
      if (payload.type === "search.started") run.searchCount += 1;
      run.activities = run.activities.slice(-20);
    } else if (event === "interaction") {
      run.status = "waiting"; run.pending = payload.kind || payload.type;
      if (run.pending === "research_plan") run.pending = "plan";
      if (run.pending === "clarification") run.phases.scope = "running";
      if (run.pending === "plan") { run.phases.scope = "completed"; run.phases.plan = "running"; }
      const duplicate = currentMessages().some((item) => item.kind === run.pending && !item.resolved);
      if (!duplicate) currentMessages().push({ id: crypto.randomUUID(), role: "assistant", kind: run.pending, data: payload, createdAt: Date.now(), resolved: false });
    } else if (event === "report" || event === "report/partial") {
      const chunk = payload.content || payload.delta || "";
      state.reports[state.currentThreadId] = payload.full_content || `${state.reports[state.currentThreadId] || ""}${chunk}`;
      upsertReportMessage(state.reports[state.currentThreadId]);
      run.phases.research = "completed"; run.phases.report = "running";
    } else if (event === "state") {
      const finalReport = payload.final_report || payload.values?.final_report;
      if (finalReport) { state.reports[state.currentThreadId] = finalReport; upsertReportMessage(finalReport); }
    } else if (event === "error") {
      run.status = "failed";
      const active = PHASES.find((p) => run.phases[p] === "running");
      if (active) run.phases[active] = "failed";
      addSystemError(payload.message || payload.detail || "研究过程中发生错误");
    } else if (event === "end") {
      run.status = payload.status === "interrupted" ? "waiting" : (payload.status || "completed");
      if (run.status === "completed") {
        PHASES.forEach((p) => { run.phases[p] = "completed"; });
        run.title = "研究已完成"; run.subtitle = "报告与来源已经整理完毕";
      }
      run.endedAt = Date.now();
    }
    persist(); render();
  }

  function upsertReportMessage(content) {
    const messages = currentMessages();
    const existing = messages.find((item) => item.kind === "report");
    if (existing) existing.content = content;
    else messages.push({ id: crypto.randomUUID(), role: "assistant", kind: "report", content, createdAt: Date.now() });
  }

  function addSystemError(content) {
    const messages = currentMessages();
    if (!messages.some((item) => item.kind === "error" && item.content === content)) {
      messages.push({ id: crypto.randomUUID(), role: "assistant", kind: "error", content, createdAt: Date.now() });
    }
  }

  function handleError(error) {
    if (error?.name === "AbortError") return;
    state.streaming = false; state.controller = null;
    const run = currentRun();
    if (run) {
      run.status = "failed";
      run.endedAt = Date.now();
      const active = PHASES.find((p) => run.phases[p] === "running");
      if (active) run.phases[active] = "failed";
    }
    addSystemError(error?.message || "无法连接研究服务");
    persist(); render();
  }

  function render() {
    renderThreads(); renderMessages(); renderHeader(); renderArtifact(); renderTaskPanel(); updateConfigSummary();
  }

  function renderThreads() {
    if (!state.threads.length) {
      els.threadList.innerHTML = '<div class="thread-empty">还没有研究记录<br>从右侧提出一个问题开始</div>';
      return;
    }
    els.threadList.innerHTML = state.threads.map((thread) => `
      <div class="thread-item ${thread.id === state.currentThreadId ? "active" : ""}" data-thread="${escapeHtml(thread.id)}">
        <span class="thread-icon">研</span><span class="thread-copy"><strong>${escapeHtml(thread.title || "未命名研究")}</strong><small>${timeLabel(thread.updatedAt || thread.createdAt)}</small></span>
      </div>`).join("");
  }

  function renderMessages() {
    const messages = currentMessages();
    els.welcome.hidden = messages.length > 0;
    const html = messages.map(renderMessage).join("");
    const run = currentRun();
    els.messageStack.innerHTML = html + (run ? renderRunCard(run) : "");
    const pendingInteraction = [...messages].reverse().find((item) =>
      ["clarification", "plan"].includes(item.kind) && !item.resolved
    );
    requestAnimationFrame(() => {
      if (pendingInteraction) {
        els.messageStack.querySelector(`[data-card="${pendingInteraction.id}"]`)?.scrollIntoView({ block: "start" });
      } else {
        els.conversation.scrollTop = els.conversation.scrollHeight;
      }
    });
  }

  function renderMessage(message) {
    if (message.kind === "clarification") return renderClarification(message);
    if (message.kind === "plan") return renderPlan(message);
    const isUser = message.role === "user";
    const body = message.kind === "error"
      ? `<strong style="color:var(--danger)">运行失败</strong><p>${escapeHtml(message.content)}</p>`
      : `<div class="md">${md(message.content || "")}</div>`;
    return `<div class="message-row ${isUser ? "user" : "assistant"}">
      <div class="avatar">${isUser ? "你" : "AI"}</div><div class="message-bubble">${body}</div></div>`;
  }

  function renderClarification(message) {
    const data = message.data || {};
    const clarification = data.clarification || data;
    const questions = clarification.questions || [];
    const options = clarification.suggested_focus || clarification.options || [];
    return `<article class="interaction-card" data-card="${message.id}">
      <div class="interaction-head"><span class="interaction-symbol">?</span><div><strong>先确认一下研究重点</strong><small>这能让检索范围和最终报告更贴近你的需求</small></div></div>
      <div class="interaction-body">
        <p class="interaction-intro">${escapeHtml(clarification.intro || "这个问题覆盖面较广，请选择你更关注的方向；也可以直接跳过，由 AI 采用综合视角。")}</p>
        ${questions.length ? `<ol class="question-list">${questions.map((q) => `<li>${escapeHtml(q)}</li>`).join("")}</ol>` : ""}
        ${options.length ? `<div class="focus-grid">${options.map((option, i) => `<label class="focus-chip"><input type="checkbox" value="${escapeHtml(option)}" id="focus-${message.id}-${i}"><span>${escapeHtml(option)}</span></label>`).join("")}</div>` : ""}
        ${message.resolved ? `<div class="resolved-note">已提交：${message.resolution === "skip" ? "采用综合视角" : "已补充研究要求"}</div>` : `
          <textarea class="feedback-input" placeholder="也可以直接输入时间范围、目标读者、重点行业或希望对比的对象…"></textarea>
          <div class="interaction-actions"><button class="secondary-button" data-action="skip" data-id="${message.id}">跳过并生成大纲</button><button class="primary-button" data-action="supplement" data-id="${message.id}">提交补充要求</button></div>`}
      </div></article>`;
  }

  function renderPlan(message) {
    const data = message.data || {};
    const plan = data.plan || data.research_plan || data;
    const sections = plan.sections || [];
    const estimated = plan.estimated_searches || currentRun()?.budget || config.max_search_calls;
    return `<article class="interaction-card" data-card="${message.id}">
      <div class="interaction-head"><span class="interaction-symbol">⌁</span><div><strong>研究方案已准备好</strong><small>确认后才会开始联网检索；不满意可以继续修改</small></div></div>
      <div class="interaction-body">
        <h3 class="plan-title">${escapeHtml(plan.title || "研究方案")}</h3>
        <p class="plan-objective">${escapeHtml(plan.objective || "将围绕核心问题收集、交叉验证信息并形成结构化报告。")}</p>
        <div class="plan-sections">${sections.map((section, index) => `<div class="plan-section"><span class="plan-number">${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(section.title || section)}</strong>${section.description ? `<p>${escapeHtml(section.description)}</p>` : ""}</div></div>`).join("")}</div>
        <div class="plan-meta"><span class="meta-pill">预计最多 ${escapeHtml(estimated)} 次搜索 / 方向</span><span class="meta-pill">${escapeHtml(config.search_apis.join(" + "))}</span><span class="meta-pill">${escapeHtml(modelDisplay())}</span></div>
        ${message.resolved ? `<div class="resolved-note">${message.resolution === "approve" ? "方案已确认，研究正在执行" : "修改要求已提交，AI 正在重拟大纲"}</div>` : `
          <div class="plan-edit" hidden><textarea class="feedback-input" placeholder="例如：缩短技术历史部分，增加国内企业案例和成本对比…"></textarea></div>
          <div class="interaction-actions"><button class="secondary-button" data-action="show-revise" data-id="${message.id}">修改方案</button><button class="primary-button" data-action="approve" data-id="${message.id}">确认并开始研究</button></div>`}
      </div></article>`;
  }

  function renderRunCard(run) {
    const completed = PHASES.filter((p) => run.phases[p] === "completed").length;
    const labels = { running: "正在执行", waiting: "等待确认", completed: "研究完成", failed: "运行失败", stopped: "已停止" };
    const active = PHASES.find((phase) => run.phases[phase] === "running");
    return `<article class="run-card compact-run">
      <span class="run-pulse ${run.status}"></span>
      <span class="run-title"><strong>${escapeHtml(run.title || "研究任务")}</strong><small>${escapeHtml(active ? PHASE_LABELS[active][1] : run.subtitle || labels[run.status])}</small></span>
      <span class="run-inline-meta">${completed}/4 · ${formatElapsed(elapsedSeconds(run))}</span>
      <button class="task-link" data-action="show-tasks">查看任务</button>
    </article>`;
  }

  function renderTaskPanel() {
    const run = currentRun();
    els.taskPanel.hidden = !run;
    if (!run) return;
    const completed = PHASES.filter((phase) => run.phases[phase] === "completed").length;
    $("taskProgress").textContent = `已完成 ${completed}/${PHASES.length}`;
    els.elapsedTime.textContent = formatElapsed(elapsedSeconds(run));
    els.taskPanelBody.innerHTML = PHASES.map((phase) => {
      const status = run.phases[phase] || "pending";
      const displayStatus = run.status === "stopped" && status === "running" ? "stopped" : status;
      const detail = displayStatus === "completed" ? "已完成" : displayStatus === "running" ? PHASE_LABELS[phase][1] : displayStatus === "failed" ? "执行失败" : displayStatus === "stopped" ? "已停止" : "等待中";
      return `<div class="task-item ${displayStatus}"><span>${displayStatus === "completed" ? "✓" : displayStatus === "failed" || displayStatus === "stopped" ? "!" : ""}</span><div><strong>${PHASE_LABELS[phase][0]}</strong><small>${detail}</small></div></div>`;
    }).join("") + (run.activities.length ? `<div class="task-current"><i></i><span><strong>${escapeHtml(run.activities.at(-1).title || run.activities.at(-1).query || "正在处理资料")}</strong><small>${escapeHtml(activityDetail(run.activities.at(-1)))}</small></span></div>` : "");
  }

  function activityDetail(item) {
    if (item.type === "search.started") return "正在检索并筛选来源";
    if (item.type === "search.completed") return "检索结果已交给研究智能体分析";
    if (item.type === "research.unit.started") return "独立研究方向已启动";
    if (item.type === "research.unit.completed") return "该方向的证据收集已经完成";
    return item.status === "completed" ? "已完成" : "正在执行";
  }

  function renderHeader() {
    const thread = state.threads.find((item) => item.id === state.currentThreadId);
    const run = currentRun();
    els.pageTitle.textContent = thread?.title || "新研究";
    els.pageSubtitle.textContent = run ? (run.subtitle || "研究任务") : "从一个好问题开始";
    const active = run && ["running", "waiting"].includes(run.status);
    els.runBadge.hidden = !active;
    if (active) els.runBadge.querySelector("b").textContent = run.status === "waiting" ? "等待确认" : "研究中";
    els.send.disabled = state.streaming;
    $("stopResearch").hidden = !(run && ["running", "waiting"].includes(run.status));
    els.showReport.hidden = !state.currentThreadId || !state.reports[state.currentThreadId];
  }

  function renderArtifact() {
    const report = state.currentThreadId ? state.reports[state.currentThreadId] : "";
    els.artifactContent.innerHTML = report ? `<div class="md">${md(report)}</div>` : '<div class="artifact-empty">报告生成后将在这里显示</div>';
  }

  function openThread(id) {
    state.currentThreadId = id; persist(); render(); closeMobileSidebar();
  }

  function newResearchView() {
    if (state.streaming) { toast("当前研究仍在运行，请等待完成"); return; }
    state.currentThreadId = null; persist(); render(); els.input.focus(); closeMobileSidebar();
  }

  function modelDisplay() {
    if (config.model_provider === "ollama") return config.model_name.replace(/^ollama:/, "Ollama · ");
    if (config.model_provider === "deepseek") return "DeepSeek";
    return config.model_name;
  }

  function updateConfigSummary() {
    const sourceNames = { tavily: "Tavily", duckduckgo: "DuckDuckGo", openai: "OpenAI Search", anthropic: "Anthropic Search" };
    $("configSummary").innerHTML = `<span class="model-dot"></span>${escapeHtml(modelDisplay())} · ${escapeHtml(config.search_apis.map((x) => sourceNames[x] || x).join(" + "))} · 最多 ${config.max_search_calls} 轮深搜`;
  }

  function applyConfigToForm() {
    $("searchBudget").value = config.max_search_calls; $("searchBudgetValue").textContent = config.max_search_calls;
    document.querySelectorAll('input[name="source"]').forEach((input) => { input.checked = config.search_apis.includes(input.value); });
    $("modelProvider").value = config.model_provider; $("modelName").value = config.model_name;
    $("ollamaBaseUrl").value = config.ollama_base_url; $("sameModel").checked = config.same_model;
    $("researchModel").value = config.research_model; $("summarizationModel").value = config.summarization_model;
    $("compressionModel").value = config.compression_model; $("reportModel").value = config.final_report_model;
    $("mcpEnabled").checked = config.mcp_enabled; $("mcpUrl").value = config.mcp_url;
    $("mcpTools").value = config.mcp_tools.join(", "); $("mcpAuth").checked = config.mcp_auth_required;
    $("mcpPrompt").value = config.mcp_prompt; $("generateReport").checked = config.generate_report;
    $("reportLanguage").value = config.report_language; $("reportLength").value = config.report_length;
    $("requireCitations").checked = config.require_citations; $("apiBase").value = config.api_base;
    renderKnowledgeList();
    toggleProviderFields(); toggleAdvancedModels();
  }

  function collectConfig() {
    const sources = [...document.querySelectorAll('input[name="source"]:checked')].map((input) => input.value);
    if (!sources.length) throw new Error("请至少选择一个搜索来源");
    const modelName = $("modelName").value.trim();
    if (!modelName) throw new Error("请填写模型标识");
    return {
      ...config, search_apis: sources, max_search_calls: Number($("searchBudget").value),
      max_researcher_iterations: Number($("searchBudget").value),
      max_react_tool_calls: Math.max(4, Number($("searchBudget").value) * 2),
      allow_clarification: false,
      model_provider: $("modelProvider").value, model_name: modelName,
      same_model: $("sameModel").checked, ollama_base_url: $("ollamaBaseUrl").value.trim(),
      research_model: $("researchModel").value.trim() || modelName,
      summarization_model: $("summarizationModel").value.trim() || modelName,
      compression_model: $("compressionModel").value.trim() || modelName,
      final_report_model: $("reportModel").value.trim() || modelName,
      mcp_enabled: $("mcpEnabled").checked,
      mcp_url: $("mcpUrl").value.trim(),
      mcp_tools: $("mcpTools").value.split(",").map((item) => item.trim()).filter(Boolean),
      mcp_auth_required: $("mcpAuth").checked,
      mcp_prompt: $("mcpPrompt").value.trim(),
      generate_report: $("generateReport").checked,
      report_language: $("reportLanguage").value,
      report_length: $("reportLength").value,
      require_citations: $("requireCitations").checked,
      api_base: $("apiBase").value.trim() || "/api",
    };
  }

  function toggleProviderFields() {
    const provider = $("modelProvider").value;
    $("ollamaSettings").hidden = provider !== "ollama";
    if (provider === "deepseek" && !$("modelName").value.startsWith("deepseek:")) $("modelName").value = "deepseek:deepseek-chat";
    if (provider === "ollama" && !$("modelName").value.startsWith("ollama:")) $("modelName").value = "ollama:";
  }

  function toggleAdvancedModels() { $("advancedModels").hidden = $("sameModel").checked; }

  function openSettings() { applyConfigToForm(); els.settings.classList.add("open"); els.drawerOverlay.classList.add("open"); }
  function closeSettings() { els.settings.classList.remove("open"); els.drawerOverlay.classList.remove("open"); }

  async function refreshOllama() {
    const status = $("ollamaStatus");
    status.className = "connection-status"; status.innerHTML = "<span></span>正在连接 Ollama…";
    try {
      const response = await fetch(api(`/providers/ollama/models?base_url=${encodeURIComponent($("ollamaBaseUrl").value.trim())}`));
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      const models = data.models || [];
      $("ollamaModel").innerHTML = models.length ? models.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("") : '<option value="">没有发现本地模型</option>';
      status.className = "connection-status connected"; status.innerHTML = `<span></span>连接成功，发现 ${models.length} 个模型`;
      if (models.length) { $("ollamaModel").value = models[0]; $("modelName").value = `ollama:${models[0]}`; }
    } catch (error) {
      status.className = "connection-status failed"; status.innerHTML = `<span></span>${escapeHtml(error.message || "无法连接 Ollama")}`;
    }
  }

  function renderKnowledgeList() {
    const list = $("knowledgeList");
    if (!config.knowledge_bases.length) { list.innerHTML = "<span>尚未导入知识文件</span>"; return; }
    list.innerHTML = config.knowledge_bases.map((item) => `<div><span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.size || "已导入")}</small></span><button type="button" data-remove-knowledge="${escapeHtml(item.id)}" aria-label="移除">×</button></div>`).join("");
  }

  async function syncKnowledgeList() {
    try {
      const response = await fetch(api("/knowledge"));
      if (!response.ok) return;
      const data = await response.json();
      config.knowledge_bases = (data.items || []).map((item) => ({
        id: item.id, name: item.name, size: `${item.chunks || 0} 个切片`,
      }));
      persist(); renderKnowledgeList();
    } catch { /* backend may not be running while viewing static frontend */ }
  }

  async function importKnowledge(files) {
    for (const file of files) {
      const content = await new Promise((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(file);
      });
      const response = await fetch(api("/knowledge"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: file.name, content, content_type: file.type }),
      });
      if (!response.ok) throw new Error(await response.text() || `无法导入 ${file.name}`);
      const item = await response.json();
      config.knowledge_bases.push({ id: item.id, name: item.name, size: `${Math.max(1, Math.round(file.size / 1024))} KB` });
    }
    persist(); renderKnowledgeList(); toast(`已导入 ${files.length} 个知识文件`);
  }

  function resizeInput() { els.input.style.height = "auto"; els.input.style.height = `${Math.min(els.input.scrollHeight, 130)}px`; }
  function openMobileSidebar() { els.sidebar.classList.add("mobile-open"); els.mobileOverlay.classList.add("open"); }
  function closeMobileSidebar() { els.sidebar.classList.remove("mobile-open"); els.mobileOverlay.classList.remove("open"); }

  function handleInteractionClick(event) {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    if (button.dataset.action === "show-tasks") {
      els.taskPanel.classList.remove("collapsed");
      $("taskPanelToggle").setAttribute("aria-expanded", "true");
      return;
    }
    if (currentRun()?.status === "stopped") { toast("该研究已停止，请发起新研究继续"); return; }
    const card = button.closest("[data-card]");
    const id = button.dataset.id;
    if (button.dataset.action === "show-revise") {
      const editor = card.querySelector(".plan-edit");
      if (editor.hidden) {
        editor.hidden = false;
        button.textContent = "提交修改";
        editor.querySelector("textarea").focus();
        return;
      }
      const feedback = editor.querySelector("textarea").value.trim();
      if (!feedback) { toast("请先填写希望如何修改"); return; }
      resumeResearch({ action: "revise", feedback }, id);
      return;
    }
    if (button.dataset.action === "approve") return resumeResearch({ action: "approve" }, id);
    if (button.dataset.action === "skip") return resumeResearch({ action: "skip" }, id);
    if (button.dataset.action === "supplement") {
      const selected = [...card.querySelectorAll('.focus-chip input:checked')].map((input) => input.value);
      const feedback = card.querySelector("textarea")?.value.trim() || "";
      if (!selected.length && !feedback) { toast("请选择关注方向或填写补充要求"); return; }
      return resumeResearch({ action: "supplement", selected, feedback }, id);
    }
  }

  function bindEvents() {
    els.send.addEventListener("click", startResearch);
    els.input.addEventListener("input", resizeInput);
    els.input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); startResearch(); } });
    els.messageStack.addEventListener("click", handleInteractionClick);
    els.threadList.addEventListener("click", (event) => { const item = event.target.closest("[data-thread]"); if (item) openThread(item.dataset.thread); });
    document.querySelectorAll(".prompt-suggestion").forEach((button) => button.addEventListener("click", () => { els.input.value = button.dataset.prompt; resizeInput(); els.input.focus(); }));
    $("newResearch").addEventListener("click", newResearchView);
    $("sidebarCollapse").addEventListener("click", () => { els.sidebar.classList.add("collapsed"); $("sidebarExpand").classList.add("visible"); });
    $("sidebarExpand").addEventListener("click", () => { els.sidebar.classList.remove("collapsed"); $("sidebarExpand").classList.remove("visible"); });
    $("mobileMenu").addEventListener("click", openMobileSidebar); els.mobileOverlay.addEventListener("click", closeMobileSidebar);
    [$("openSettings"), $("headerSettings"), $("configSummary")].forEach((el) => el.addEventListener("click", openSettings));
    $("closeSettings").addEventListener("click", closeSettings); els.drawerOverlay.addEventListener("click", closeSettings);
    $("searchBudget").addEventListener("input", (event) => { $("searchBudgetValue").textContent = event.target.value; });
    $("modelProvider").addEventListener("change", toggleProviderFields); $("sameModel").addEventListener("change", toggleAdvancedModels);
    $("ollamaModel").addEventListener("change", (event) => { if (event.target.value) $("modelName").value = `ollama:${event.target.value}`; });
    $("refreshOllama").addEventListener("click", refreshOllama);
    $("stopResearch").addEventListener("click", stopResearch);
    $("taskPanelToggle").addEventListener("click", () => {
      const collapsed = els.taskPanel.classList.toggle("collapsed");
      $("taskPanelToggle").setAttribute("aria-expanded", String(!collapsed));
    });
    $("importKnowledge").addEventListener("click", () => $("knowledgeFiles").click());
    $("knowledgeFiles").addEventListener("change", async (event) => {
      try { await importKnowledge([...event.target.files]); } catch (error) { toast(error.message); }
      event.target.value = "";
    });
    $("knowledgeList").addEventListener("click", async (event) => {
      const button = event.target.closest("[data-remove-knowledge]"); if (!button) return;
      const id = button.dataset.removeKnowledge;
      try { await fetch(api(`/knowledge/${encodeURIComponent(id)}`), { method: "DELETE" }); } catch { /* remove local reference anyway */ }
      config.knowledge_bases = config.knowledge_bases.filter((item) => item.id !== id); persist(); renderKnowledgeList();
    });
    $("saveSettings").addEventListener("click", () => { try { config = collectConfig(); persist(); updateConfigSummary(); applyTheme(); closeSettings(); toast("研究设置已保存"); } catch (error) { toast(error.message); } });
    $("resetSettings").addEventListener("click", () => { config = structuredClone(DEFAULT_CONFIG); applyConfigToForm(); toast("已恢复默认设置，保存后生效"); });
    $("themeToggle").addEventListener("click", () => { config.theme = config.theme === "dark" ? "light" : "dark"; applyTheme(); persist(); });
    $("showReport").addEventListener("click", () => els.artifactPanel.classList.add("open"));
    $("closeReport").addEventListener("click", () => els.artifactPanel.classList.remove("open"));
    $("copyReport").addEventListener("click", async () => { const report = state.reports[state.currentThreadId] || ""; await navigator.clipboard.writeText(report); toast("报告已复制"); });
    $("downloadReport").addEventListener("click", () => { const report = state.reports[state.currentThreadId] || ""; const blob = new Blob([report], { type: "text/markdown;charset=utf-8" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${state.threads.find((t) => t.id === state.currentThreadId)?.title || "研究报告"}.md`; a.click(); URL.revokeObjectURL(a.href); });
  }

  function applyTheme() {
    document.documentElement.dataset.theme = config.theme;
    $("themeGlyph").textContent = config.theme === "dark" ? "☀" : "☾";
    $("themeLabel").textContent = config.theme === "dark" ? "浅色模式" : "深色模式";
  }

  applyTheme(); applyConfigToForm(); bindEvents(); render(); resizeInput(); syncKnowledgeList();
})();
