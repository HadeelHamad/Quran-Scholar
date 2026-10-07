const form = document.getElementById("research-form");
const questionEl = document.getElementById("question");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultSection = document.getElementById("result-section");
const reportEl = document.getElementById("report");
const traceLogEl = document.getElementById("trace-log");
const traceDetails = document.getElementById("trace-details");
const badgesEl = document.getElementById("badges");
const diagnosticsEl = document.getElementById("diagnostics");

function setLoading(loading) {
  submitBtn.disabled = loading;
  statusEl.textContent = loading
    ? "جاري البحث… قد يستغرق دقيقة أو أكثر"
    : "";
  statusEl.classList.toggle("loading", loading);
}

function renderBadges(data) {
  badgesEl.innerHTML = "";
  if (data.research_complete) {
    badgesEl.appendChild(makeBadge("اكتمل البحث", "ok"));
  }
}

function makeBadge(text, kind) {
  const span = document.createElement("span");
  span.className = `badge ${kind}`;
  span.textContent = text;
  return span;
}

function renderDiagnostics(errors, warnings) {
  diagnosticsEl.innerHTML = "";
  const has = (errors && errors.length) || (warnings && warnings.length);
  diagnosticsEl.classList.toggle("hidden", !has);
  if (!has) return;

  if (errors?.length) {
    const div = document.createElement("div");
    div.className = "err";
    div.textContent = "أخطاء: " + errors.join(" · ");
    diagnosticsEl.appendChild(div);
  }
  if (warnings?.length) {
    const div = document.createElement("div");
    div.className = "warn";
    div.textContent = "تحذيرات: " + warnings.join(" · ");
    diagnosticsEl.appendChild(div);
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionEl.value.trim();
  if (!question) return;

  setLoading(true);
  resultSection.classList.add("hidden");

  try {
    const res = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      const detail = data.detail || res.statusText || "خطأ غير معروف";
      reportEl.textContent =
        typeof detail === "string" ? detail : JSON.stringify(detail);
      traceLogEl.textContent = "";
      badgesEl.innerHTML = "";
      diagnosticsEl.classList.add("hidden");
      resultSection.classList.remove("hidden");
      return;
    }

    reportEl.textContent =
      data.report ||
      "لم يُنتَج تقرير. راجع سجل التتبّع أو تحذيرات النظام أدناه.";
    traceLogEl.textContent = (data.trace_log || []).join("\n");
    traceDetails.open = false;
    renderBadges(data);
    renderDiagnostics(data.errors, data.warnings);
    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    reportEl.textContent = "تعذّر الاتصال بالخادم. تأكد أن الخادم يعمل.";
    traceLogEl.textContent = String(err);
    resultSection.classList.remove("hidden");
  } finally {
    setLoading(false);
  }
});
