// dashboard.js

// --- Global Config ---
// In a real app, these would come from the sidebar state

function isValidUserStory(story) {
  const lower = story.toLowerCase().trim();
  // Must start with "as a"
  if (!lower.startsWith('as a')) return false;
  // Must contain "i want" or "i need"
  if (!lower.includes('i want') && !lower.includes('i need')) return false;
  // Must have content after "i want" or "i need"
  const wantIndex = lower.indexOf('i want');
  const needIndex = lower.indexOf('i need');
  const index = wantIndex !== -1 ? wantIndex : needIndex;
  const phrase = wantIndex !== -1 ? 'i want' : 'i need';
  const after = lower.substring(index + phrase.length).trim();
  if (after.length < 3) return false; // Require at least 3 characters after the phrase
  return true;
}

function getSelectedModel() {
  return document.getElementById("modelSelect")
    ? document.getElementById("modelSelect").value
    : "ChatGPT";
}

function getSelectedStrategy() {
  return document.getElementById("strategySelect")
    ? document.getElementById("strategySelect").value
    : "Zero-shot";
}

// Helper to show/hide error banner
function showError(message) {
  const banner = document.getElementById("errorBanner");
  const bannerText = document.getElementById("errorBannerText");
  if (banner && bannerText) {
    bannerText.textContent = message;
    banner.classList.remove("hidden");
  }
}

function clearError() {
  const banner = document.getElementById("errorBanner");
  if (banner) banner.classList.add("hidden");
}

// --- Single Classification ---
async function classifyStory() {
  const input = document.getElementById("storyInput");
  const resultArea = document.getElementById("resultArea");
  const indicator = document.getElementById("statusIndicator");

  if (!input || !input.value.trim()) {
    alert("Please enter a user story.");
    return;
  }

  // Validate user story format
  if (!isValidUserStory(input.value.trim())) {
    showError("Please enter a valid user story (e.g., 'As a user, I want...').");
    return;
  }

  clearError();

  // UI Updates
  if (indicator) indicator.classList.remove("hidden");
  if (resultArea) {
    resultArea.classList.add("hidden");
    resultArea.classList.remove("opacity-100");
    resultArea.classList.add("opacity-0");
  }
  // ✅ Clear any previous error

  try {
    const response = await fetch("/single", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        story: input.value,
        model: getSelectedModel(),
        strategy: getSelectedStrategy(),
      }),
    });
    const data = await response.json();
    window.currentResultId = data.result_id;

    // ✅ Also handle other errors
    if (!response.ok) {
      // showError(data.error || "An error occurred.");
      const errMsg = typeof data.error === 'object' ? (data.error.error || JSON.stringify(data.error)) : data.error;
      showError(errMsg || "An error occurred.");
      return;
    }

    // Update Result Cards
    const badge = document.getElementById("resBadge");
    if (badge) {
      badge.textContent = data.classification;
      badge.className = `text-2xl font-bold px-4 py-2 rounded-lg border ${data.classification === "FR" ? "badge-fr" : "badge-nfr"}`;
    }

    document.getElementById("resCategory").textContent =
      data.category || "Functional Requirement";
      highlightTaxonomy(data.category);
    document.getElementById("resModel").textContent = data.model;
    document.getElementById("resStrategy").textContent = data.strategy;
    document.getElementById("resLatency").textContent = `${data.latency}s`;
    document.getElementById("resReason").innerText = data.reason || "--";
    const highlightBox = document.getElementById("highlightedStory");
if (highlightBox) {
  highlightBox.innerHTML =
    data.highlighted_story || input.value;
}

    // Step-by-Step Reasoning
    const stepByStepContainer = document.getElementById("resStepByStepContainer");
    const stepByStepEl = document.getElementById("resStepByStep");
    if (stepByStepContainer && stepByStepEl) {
      if (data.step_by_step && data.step_by_step.trim() !== "") {
        // Format the 5 steps to make them bold and readable
        let formattedReasoning = data.step_by_step
          .replace(/(Step 1:.*?)(?=\n|$)/g, '<strong class="text-blue-400">$1</strong>')
          .replace(/(Step 2:.*?)(?=\n|$)/g, '<strong class="text-blue-400">$1</strong>')
          .replace(/(Step 3:.*?)(?=\n|$)/g, '<strong class="text-blue-400">$1</strong>')
          .replace(/(Step 4:.*?)(?=\n|$)/g, '<strong class="text-blue-400">$1</strong>')
          .replace(/(Step 5:.*?)(?=\n|$)/g, '<strong class="text-blue-400">$1</strong>');
        
        stepByStepEl.innerHTML = formattedReasoning;
        stepByStepContainer.classList.remove("hidden");
      } else {
        stepByStepContainer.classList.add("hidden");
        stepByStepEl.innerHTML = "--";
      }
    }

    const confidence = data.confidence || 50;
    const confEl = document.getElementById("resConfidence");
    if (confEl) {
      const color =
        confidence >= 75
          ? "bg-green-500"
          : confidence >= 50
            ? "bg-yellow-500"
            : "bg-red-500";
      const textColor =
        confidence >= 75
          ? "text-green-400"
          : confidence >= 50
            ? "text-yellow-400"
            : "text-red-400";
      confEl.innerHTML = `
        <div class="flex items-center justify-between mb-2">
            <span class="text-xl font-bold ${textColor}">${confidence}%</span>
        </div>
        <div class="w-full bg-slate-700 rounded-full h-2">
            <div class="${color} h-2 rounded-full transition-all duration-700" style="width: ${confidence}%"></div>
        </div>
    `;
    }

    // Circle Animation removed

    // Show Results
    if (resultArea) {
      resultArea.classList.remove("hidden");
      // Small delay to allow display:block to apply before opacity transition
      const feedbackSection = document.getElementById("feedbackSection");
  if (feedbackSection) {
    feedbackSection.classList.remove("hidden");
  }

      setTimeout(() => {
        resultArea.classList.remove("opacity-0");
        resultArea.classList.add("opacity-100");
      }, 50);
    }
  } catch (e) {
    console.error(e);
    // alert("Error classifying story.");
    showError("Network error. Please check your connection.");
  } finally {
    if (indicator) indicator.classList.add("hidden");
  }
}

function clearSingleClassification() {
  const storyInput = document.getElementById("storyInput");
  const resultArea = document.getElementById("resultArea");
  const errorBanner = document.getElementById("errorBanner");
  const loadingArea = document.getElementById("loadingArea");

  if (storyInput) storyInput.value = "";
  if (resultArea) {
    resultArea.classList.add("hidden");
    resultArea.classList.remove("opacity-100");
    resultArea.classList.add("opacity-0");
  }
  if (errorBanner) errorBanner.classList.add("hidden");
  if (loadingArea) loadingArea.classList.add("hidden");
}
document.addEventListener("mouseover", function (e) {

  if (e.target.classList.contains("keyword-highlight")) {

    const tooltip = document.getElementById("keywordTooltip");
    const explanation = e.target.dataset.explanation;

    tooltip.textContent = explanation;
    tooltip.classList.remove("hidden");

    document.addEventListener("mousemove", moveTooltip);

    function moveTooltip(event) {
      tooltip.style.top = event.pageY + 15 + "px";
      tooltip.style.left = event.pageX + 15 + "px";
    }

    e.target.addEventListener("mouseleave", () => {
      tooltip.classList.add("hidden");
      document.removeEventListener("mousemove", moveTooltip);
    });
  }

});
// =========================
// TAXONOMY CATEGORY HIGHLIGHT
// =========================

function highlightTaxonomy(category){

  // remove old highlights
  document.querySelectorAll(".taxonomy-item")
    .forEach(el => el.classList.remove("taxonomy-active"));

  if(!category) return;

  const cleanCategory = category.toLowerCase().trim();
const id = "taxonomy-" + cleanCategory;
  const item = document.getElementById(id);

  if(item){
    item.classList.add("taxonomy-active");

    // 👇 NEW: scroll into view
    item.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // 👇 NEW: open panel automatically
  const panel = document.getElementById("taxonomyPanel");
if(panel && panel.classList.contains("translate-x-full")){
  panel.classList.remove("translate-x-full");
}

  // 👇 NEW: populate Q&A
  if(category && category !== "--"){
    

    // Auto ask when classification happens
  const input = document.getElementById("nfrQuestion");

if (input) {
  const hasStory = document.getElementById("storyInput");

  if (hasStory) {
    // Dashboard page
    input.value = `Why is this requirement classified as ${category}? Explain with reasoning.`;
  } else {
    // Home page
    input.value = `Explain ${category} in simple terms with examples.`;
  }

  answerQuestion();
}
  }
}
async function answerQuestion() {

  const questionInput = document.getElementById("nfrQuestion");
  const answerDiv = document.getElementById("nfrAnswer");

  const question = questionInput.value.trim();
  if(!question) return;

  answerDiv.innerHTML = `
  <div class="flex items-center gap-2 text-gray-400">
    <div class="w-3 h-3 bg-blue-500 rounded-full animate-bounce"></div>
    <div class="w-3 h-3 bg-blue-500 rounded-full animate-bounce delay-100"></div>
    <div class="w-3 h-3 bg-blue-500 rounded-full animate-bounce delay-200"></div>
    <span>Thinking...</span>
  </div>
`;

  // get current context
  const categoryEl = document.getElementById("resCategory");
const storyEl = document.getElementById("storyInput");

const category = categoryEl ? categoryEl.innerText.trim() : "";
const story = storyEl ? storyEl.value : "";
// ✅ CREATE PAYLOAD FIRST
let payload = {
  question: question
};

if (category) payload.category = category;
if (story) payload.story = story;

  try {
    const response = await fetch("/ask-nfr", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    answerDiv.innerHTML = `
  <div class="bg-white dark:bg-slate-800 p-3 rounded-lg border border-gray-200 dark:border-slate-700 shadow-sm">
    ${data.answer}
  </div>
`;

  } catch (err) {
    answerDiv.innerHTML = "⚠️ Error getting answer.";
  }
}
function toggleTaxonomyPanel(){

  const panel = document.getElementById("taxonomyPanel");

  if(panel.classList.contains("translate-x-full")){
      panel.classList.remove("translate-x-full");

      setTimeout(() => {
        const input = document.getElementById("nfrQuestion");
        if(input) input.focus();
      }, 300);

  }else{
      panel.classList.add("translate-x-full");
  }

}
function showSuggestions(value) {
  const box = document.getElementById("nfrSuggestions");
  const category = document.getElementById("resCategory")?.innerText || "";

  if (!box) return;

  const input = value.toLowerCase().trim();

  // Base suggestions
  let suggestions = [
    `Why is this requirement classified as ${category}?`,
    `How can this ${category} requirement be improved?`,
    `What are examples of ${category}?`,
    `What metrics define ${category}?`,
    `Explain ${category} in simple terms`
  ];

  // Extra smart suggestions based on keywords
  if (input.includes("why")) {
    suggestions.unshift(`Why is this classified as ${category}?`);
  }
  if (input.includes("how")) {
    suggestions.unshift(`How to improve ${category}?`);
  }
  if (input.includes("example")) {
    suggestions.unshift(`Give real-world examples of ${category}`);
  }

  // Filter suggestions
  const filtered = suggestions.filter(s =>
    s.toLowerCase().includes(input)
  );

  // Render suggestions
  box.innerHTML = filtered.slice(0, 5).map(s => `
    <div class="cursor-pointer px-2 py-1 rounded hover:bg-blue-100 dark:hover:bg-blue-900"
         onclick="selectSuggestion('${s.replace(/'/g, "\\'")}')">
      ${s}
    </div>
  `).join("");
}
function selectSuggestion(text) {
  const input = document.getElementById("nfrQuestion");
  const box = document.getElementById("nfrSuggestions");

  if (input) input.value = text;
  if (box) box.innerHTML = "";

  // 🔥 Trigger answer automatically
  answerQuestion();
}

// --- File Upload Logic ---
// Wrapped in DOMContentLoaded so DOM elements are guaranteed to exist.
document.addEventListener('DOMContentLoaded', () => {
  const dropZone        = document.getElementById("dropZone");
  const fileInput       = document.getElementById("fileInput");
  const uploadPrompt    = document.getElementById("uploadPrompt");
  const fileActionBar   = document.getElementById("fileActionBar");
  const uploadedFilename= document.getElementById("uploadedFilename");
  const removeBtn       = document.getElementById("removeBtn");
  const previewBtn      = document.getElementById("previewBtn");
  const input = document.getElementById("nfrQuestion");
if(input){
  input.addEventListener("keypress", function(e){
    if(e.key === "Enter"){
      answerQuestion();
    }
  });
}
  if (dropZone && fileInput) {
    // Only trigger file picker if clicking on dropZone BUT NOT if clicking actions
    dropZone.addEventListener("click", (e) => {
      if (!e.target.closest("button")) {
        fileInput.click();
      }
    });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("border-blue-500", "bg-slate-800/10");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("border-blue-500", "bg-slate-800/10");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("border-blue-500", "bg-slate-800/10");

      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) {
        handleFileSelect(fileInput.files[0]);
      }
    });

    if (removeBtn) {
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation(); // Stop bubble to dropZone
        removeFile();
      });
    }

    if (previewBtn) {
      previewBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        previewFile();
      });
    }
  }
});

function handleFileSelect(file) {
  const uploadedFilename = document.getElementById("uploadedFilename");
  const uploadPrompt     = document.getElementById("uploadPrompt");
  const fileActionBar    = document.getElementById("fileActionBar");

  if (uploadedFilename) uploadedFilename.textContent = file.name;
  if (uploadPrompt)
    uploadPrompt.classList.add("opacity-0", "pointer-events-none");
  if (fileActionBar) fileActionBar.classList.remove("hidden");

  // Parse CSV to get total rows AND populate window.batchStories for ambiguity scan
  const reader = new FileReader();
  reader.onload = function (e) {
    const text = e.target.result;
    const lines = text.split("\n").filter((r) => r.trim());
    const storyCount = Math.max(0, lines.length - 1);

    // ── Parse CSV columns and build window.batchStories ──────────
    function parseCSVLine(line) {
      const result = [];
      let startValueIndex = 0;
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        if (line[i] === '"') {
          if (i + 1 < line.length && line[i + 1] === '"') { i++; }
          else { inQuotes = !inQuotes; }
        } else if (line[i] === "," && !inQuotes) {
          let val = line.substring(startValueIndex, i).trim();
          if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1,-1).replace(/""/g,'"');
          result.push(val);
          startValueIndex = i + 1;
        }
      }
      let val = line.substring(startValueIndex).trim();
      if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1,-1).replace(/""/g,'"');
      result.push(val);
      return result;
    }

    if (lines.length > 1) {
      const headers = parseCSVLine(lines[0]).map(h => h.toLowerCase().trim());
      // Find the user_story column index; fall back to first column
      const storyColIdx = headers.findIndex(h =>
        h === "user_story" || h === "story" || h === "requirement" || h === "text"
      );
      const colIdx = storyColIdx >= 0 ? storyColIdx : 0;

      const rows = lines.slice(1).map(line => {
        const cells = parseCSVLine(line);
        return headers.reduce((obj, h, i) => { obj[h] = cells[i] || ""; return obj; }, {});
      });

      // Expose both formats so batch.html script can find stories
      window.uploadedCSVRows = rows;
      window.batchStories = rows.map(r => r[headers[colIdx]] || Object.values(r)[0] || "").filter(Boolean);
    } else {
      window.uploadedCSVRows = [];
      window.batchStories = [];
    }
    // ─────────────────────────────────────────────────────────────

    const simText     = document.getElementById("simulateContainerText");
    const simSlider   = document.getElementById("simulateContainerSlider");
    const volumeSlider= document.getElementById("volumeSlider");
    const volumeValue = document.getElementById("volumeValue");

    if (volumeSlider) {
        volumeSlider.max   = storyCount;
        volumeSlider.value = Math.min(volumeSlider.value, storyCount);
        if (volumeValue) volumeValue.textContent = volumeSlider.value;
    }
    if (simText) {
        simText.querySelector("label").textContent = "Select Stories to Process";
        simText.querySelector("p").textContent = `Found ${storyCount} stories in CSV. Choose how many to process.`;
        simText.classList.remove("hidden");
    }
    if (simSlider) simSlider.classList.remove("hidden");
  };
  reader.readAsText(file);
}

function removeFile() {
  const fileInput     = document.getElementById("fileInput");
  const uploadPrompt  = document.getElementById("uploadPrompt");
  const fileActionBar = document.getElementById("fileActionBar");

  if (fileInput)    fileInput.value = "";
  if (uploadPrompt) uploadPrompt.classList.remove("opacity-0", "pointer-events-none");
  if (fileActionBar) fileActionBar.classList.add("hidden");
  window.batchStories = [];
  window.uploadedCSVRows = [];

  const simText     = document.getElementById("simulateContainerText");
  const simSlider   = document.getElementById("simulateContainerSlider");
  const volumeSlider= document.getElementById("volumeSlider");
  const volumeValue = document.getElementById("volumeValue");

  if (volumeSlider) {
      volumeSlider.max   = 50;
      volumeSlider.value = Math.min(volumeSlider.value, 50);
      if (volumeValue) volumeValue.textContent = volumeSlider.value;
  }
  if (simText) {
      simText.querySelector("label").textContent = "Simulate Volume";
      simText.querySelector("p").textContent = "For demo purposes, choose how many dummy stories to generate.";
      simText.classList.add("hidden");
  }
  if (simSlider) simSlider.classList.add("hidden");
}

function previewFile() {
  const fileInput = document.getElementById("fileInput");
  const file = fileInput ? fileInput.files[0] : null;
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    const text = e.target.result;
    const rows = text.split("\n").slice(0, 101); // Top 100 rows + header

    // Helper to parse CSV line respecting quotes
    function parseCSVLine(line) {
      const result = [];
      let startValueIndex = 0;
      let inQuotes = false;

      for (let i = 0; i < line.length; i++) {
        if (line[i] === '"') {
          // Check for escaped quote ""
          if (i + 1 < line.length && line[i + 1] === '"') {
            i++; // Skip next quote
          } else {
            inQuotes = !inQuotes;
          }
        } else if (line[i] === "," && !inQuotes) {
          let val = line.substring(startValueIndex, i);
          // Remove wrapping quotes if present
          val = val.trim();
          if (val.startsWith('"') && val.endsWith('"')) {
            val = val.slice(1, -1).replace(/""/g, '"');
          }
          result.push(val);
          startValueIndex = i + 1;
        }
      }

      // Push last field
      let val = line.substring(startValueIndex);
      val = val.trim();
      if (val.startsWith('"') && val.endsWith('"')) {
        val = val.slice(1, -1).replace(/""/g, '"');
      }
      result.push(val);
      return result;
    }

    if (rows.length === 0) return;

    const header = parseCSVLine(rows[0]);
    const bodyRows = rows.slice(1).filter((r) => r.trim());

    const headerRowEl = document.getElementById("previewHeaderRow");
    const bodyEl = document.getElementById("previewBody");

    if (headerRowEl) {
      headerRowEl.innerHTML = header
        .map(
          (h) =>
            `<th class="px-6 py-3 bg-slate-800 sticky top-0 z-10 font-bold text-gray-200 border-b border-slate-700 shadow-sm text-xs uppercase tracking-wider whitespace-nowrap">${h}</th>`,
        )
        .join("");
    }
    if (bodyEl) {
      bodyEl.innerHTML = bodyRows
        .map((row) => {
          const cells = parseCSVLine(row);
          return `<tr class="border-b border-slate-700/50 hover:bg-slate-800/50 transition-colors">${cells.map((c) => `<td class="px-6 py-3 whitespace-nowrap">${c}</td>`).join("")}</tr>`;
        })
        .join("");
    }

    document.getElementById("previewModal").classList.remove("hidden");
  };
  reader.readAsText(file);
}

function closePreview() {
  document.getElementById("previewModal").classList.add("hidden");
}

// --- Batch Processing ---
let isBatchFrPlotly = false; // true when the FR chart is rendered via Plotly
let batchCatChart   = null;
let batchResultsData = []; // Store results for export

function setupBatchCharts() {
  // ... (Chart setup code remains same, omitted for brevity if possible, but replace tool needs context.
  // Since I'm replacing a large chunk or just adding variables, let's keep it safe.)
  Chart.defaults.color = "#94a3b8";
  Chart.defaults.borderColor = "#334155";

  // FR vs NFR (Sunburst using Plotly)
  const ctx1 = document.getElementById("batchFrChart");
  if (ctx1) {
    // Clear out any existing Chart.js canvas if it exists inside the container
    // Plotly needs a div, not a canvas, so we will replace the canvas with a div or just draw on the parent container.
    // For safety, let's create a new div inside the parent if it's currently a canvas, or just use the parent element.
    const parent = ctx1.parentElement;
    parent.innerHTML = '<div id="batchFrPlotly" style="width:100%; height:100%;"></div>';
    
    // Initial empty data for the Sunburst chart
    const data = [{
      type: "sunburst",
      labels: ["Total", "Functional", "Non-Functional"],
      parents: ["", "Total", "Total"],
      values: [0, 0, 0],
      branchvalues: "total",
      marker: {
        colors: ["#1e293b", "#10b981", "#ef4444"]
      },
      textinfo: "label+value+percent parent",
      hoverinfo: "label+value+percent root",
      insidetextfont: { family: "'Inter', sans-serif", size: 14, color: "#ffffff" },
    }];

    const layout = {
      margin: { l: 0, r: 0, b: 0, t: 0 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent"
    };

    Plotly.newPlot('batchFrPlotly', data, layout, {responsive: true, displayModeBar: false});
    isBatchFrPlotly = true; // mark that the FR chart is using Plotly
  }  // Categories (Premium Radar Chart)
  const ctx2 = document.getElementById("batchCatChart");
  if (ctx2) {
    if (batchCatChart) batchCatChart.destroy();
    batchCatChart = new Chart(ctx2, {
      type: "radar",
      data: {
        labels: [
          "Accuracy", "Usability", "Performance", "Efficiency", "Security",
          "Privacy", "Fairness & Bias", "Explainability", "Interpretability",
          "Transparency", "Accessibility", "Reliability", "Robustness",
          "Maintainability", "Scalability", "Interoperability",
          "Completeness & Consistency", "Trust", "Safety & Governance"
        ],
        datasets: [{
          label: "Frequency",
          data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          backgroundColor: "rgba(139, 92, 246, 0.25)",
          borderColor: "#8b5cf6",
          pointBackgroundColor: "#c4b5fd",
          pointBorderColor: "#fff",
          pointHoverBackgroundColor: "#fff",
          pointHoverBorderColor: "#8b5cf6",
          borderWidth: 2,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        elements: {
          line: { tension: 0.3 } // slight curve to the lines
        },
        plugins: { 
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(15, 23, 42, 0.9)",
            titleColor: "#f8fafc",
            bodyColor: "#cbd5e1",
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1,
            padding: 10,
            displayColors: false
          }
        },
        scales: {
          r: {
            angleLines: { color: "rgba(255, 255, 255, 0.06)" },
            grid: { color: "rgba(255, 255, 255, 0.06)", circular: true },
            pointLabels: {
              color: "#94a3b8",
              font: { family: "'Inter', sans-serif", size: 10, weight: '500' }
            },
            ticks: {
              display: false, // hide inner numbers for a cleaner look
              stepSize: 1,
              beginAtZero: true
            }
          }
        }
      }
    });
  }
}

let currentMode = "normal"; // default mode
async function processBatch() {
  const tableBody = document.getElementById("resultsTableBody");
  const batchResults = document.getElementById("batchResults");
  const volume = document.getElementById("volumeSlider").value;
  const fileInput = document.getElementById("fileInput");

if (!fileInput || fileInput.files.length === 0) {
  showBatchError("Please upload a CSV file first.");
  return;
}

// clear error if valid
clearBatchError();
  const exportBtn = document.getElementById("exportBtn");
  // Reset server-side batch data before starting new batch
  await fetch("api/reset_batch", {method: "POST"}); 

  if (tableBody)
    tableBody.innerHTML =
      '<tr><td colspan="5" class="px-6 py-4 text-center text-gray-500">Processing...</td></tr>';
  if (batchResults) batchResults.classList.remove("hidden");

  // Reset data
  batchResultsData = [];
  if (exportBtn) exportBtn.disabled = true;

  setupBatchCharts();

  const formData = new FormData();
  formData.append("count", volume);
  formData.append("model", getSelectedModel());
  formData.append("strategy", getSelectedStrategy());
  formData.append("mode", currentMode);

  if (fileInput && fileInput.files.length > 0) {
    formData.append("file", fileInput.files[0]);
  }

  try {
    const response = await fetch("/batch", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      try {
        const errorJson = JSON.parse(errorText);
        throw new Error(errorJson.error || "Unknown error occurred");
      } catch (e) {
        // If not JSON, use text or status text
        throw new Error(errorText || response.statusText);
      }
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");

      // Process all complete lines
      for (let i = 0; i < lines.length - 1; i++) {
        const line = lines[i].trim();
        if (line) {
          try {
            const data = JSON.parse(line);
            handleStreamData(data);
          } catch (e) {
            console.error("Error parsing JSON line:", e);
          }
        }
      }
      // Keep the last partial line in buffer
      buffer = lines[lines.length - 1];
    }

    // Process any remaining buffer
    if (buffer.trim()) {
      try {
        const data = JSON.parse(buffer);
        handleStreamData(data);
      } catch (e) {
        console.error("Error parsing final JSON:", e);
      }
    }
  } catch (e) {
    console.error(e);
  }

}

function showBatchError(message) {
  const banner = document.getElementById("batchErrorBanner");
  const text = document.getElementById("batchErrorText");

  if (banner && text) {
    text.textContent = message;
    banner.classList.remove("hidden");
  }
}

function clearBatchError() {
  const banner = document.getElementById("batchErrorBanner");
  if (banner) banner.classList.add("hidden");
}
function handleStreamData(data) {
  
// console.log("STREAM DATA:", data);
  // =========================
  // 🔥 1. HANDLE GROUP START
  // =========================
  if (data.type === "group_start") {
    const tableBody = document.getElementById("resultsTableBody");

    const row = document.createElement("tr");
    row.id = `group-${data.group_id}`;

    row.innerHTML = `
      <td colspan="5" class="px-6 py-3 bg-blue-100 dark:bg-blue-900/30 font-bold">
        📦 Group ${data.group_id + 1} (${data.size} items)
      </td>
    `;

    tableBody.appendChild(row);
    return; // ✅ IMPORTANT
  }


  // =========================
  // ✅ EXISTING RESULT LOGIC (MODIFIED)
  // =========================
  if (data.type === "result") {

    // Collect data for export
    batchResultsData.push({
      story: data.story,
      classification: data.result.classification,
      category: data.result.category || "",
      latency: data.result.latency,
      reason: data.result.reason || "",
    });

    // Enable export button
    // Enable export buttons if we have data
    const exportBtn = document.getElementById("exportBtn");
    const exportBtnBottom = document.getElementById("exportBtnBottom");
    if (exportBtn && batchResultsData.length > 0) exportBtn.disabled = false;
    if (exportBtnBottom && batchResultsData.length > 0) exportBtnBottom.disabled = false;

    // =========================
    // 🔥 2. SAFE STATS (FIX CRASH)
    // =========================
    if (data.current_stats) {
      const stats = data.current_stats;
// console.log("STATS RECEIVED:", stats); 
      document.getElementById("totalCount").textContent = stats.total;
      document.getElementById("frCount").textContent = stats.fr_count;
      document.getElementById("nfrCount").textContent = stats.nfr_count;
      document.getElementById("avgTime").textContent = `${stats.avg_time}s`;

      if (isBatchFrPlotly && stats.total > 0) {
        Plotly.update('batchFrPlotly', {
          labels: [["Total", "Functional", "Non-Functional"]],
          parents: [["", "Total", "Total"]],
          values: [[stats.total, stats.fr_count, stats.nfr_count]]
        });
      }

      if (batchCatChart && stats.category_counts) {
        const labels = batchCatChart.data.labels;
        const newData = labels.map((l) => stats.category_counts[l] || 0);
        batchCatChart.data.datasets[0].data = newData;
        batchCatChart.update("none");
      }
    }

    // =========================
    // TABLE ROW
    // =========================
    const tableBody = document.getElementById("resultsTableBody");

    if (tableBody) {
      if (tableBody.querySelector('td[colspan="4"]')) {
        tableBody.innerHTML = "";
      }

      const item = data;
      const row  = document.createElement("tr");
      row.className = "fade-in";

      const isFR = item.result.classification === "FR";

      row.innerHTML = `
        <td class="px-6 py-4 font-medium text-gray-900 dark:text-white">${item.story}</td>
        <td class="px-6 py-4">
          <span class="${isFR ? "text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30" : "text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30"} px-2 py-1 rounded text-xs font-bold border ${isFR ? "border-green-200 dark:border-green-500/30" : "border-red-200 dark:border-red-500/30"}">
            ${item.result.classification}
          </span>
        </td>
        <td class="px-6 py-4 text-gray-500 dark:text-gray-400">${item.result.category || "-"}</td>
        <td class="px-6 py-4 text-gray-500 dark:text-gray-400 text-sm">${item.result.reason || "-"}</td>
        <td class="px-6 py-4 font-mono text-gray-500">${item.result.latency}s</td>
      `;

      // =========================
      // 🔥 3. INSERT UNDER GROUP
      // =========================
      const groupRow = data.group_id !== undefined
        ? document.getElementById(`group-${data.group_id}`)
        : null;

      if (groupRow) {
        groupRow.insertAdjacentElement("afterend", row);
      } else {
        tableBody.appendChild(row);
      }
    }
  }


  // =========================
  // EXISTING STOP LOGIC
  // =========================
  else if (data.type === "stopped") {
    const submitBtn = document.getElementById("submitBtn");
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = "Start Processing";
    }

    const tableBody = document.getElementById("resultsTableBody");
    if (tableBody && batchResultsData.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="5" class="px-6 py-8 text-center text-gray-400">
        Processing stopped.
      </td></tr>`;
    }

    const exportBtn = document.getElementById("exportBtn");
    if (exportBtn && batchResultsData.length > 0) exportBtn.disabled = false;
  }


  // =========================
  // EXISTING GROUP DISPLAY (KEEP)
  // =========================
  else if (data.type === "groups") {
    console.log("Groups received:", data.groups);

    const container = document.getElementById("groupsContainer");
    if (container) {
      container.classList.remove("hidden");
    }

    displayGroups(data.groups);
  }
}
function displayGroups(groups) {
  const list = document.getElementById("groupsList");
  if (!list) return;

  list.innerHTML = "";

  Object.keys(groups).forEach(groupName => {
    const groupDiv = document.createElement("div");
    groupDiv.className = "mb-4 p-4 border rounded";

    groupDiv.innerHTML = `
      <h3 class="font-bold text-lg mb-2">${groupName}</h3>
      <ul class="list-disc ml-5">
        ${groups[groupName]
          .map(story => `<li>${story}</li>`)
          .join("")}
      </ul>
    `;

    list.appendChild(groupDiv);
  });
}

function exportBatchCSV() {
  if (!batchResultsData || batchResultsData.length === 0) return;

  // CSV Header
  const headers = [
    "User Story",
    "Classification",
    "Category",
    "Latency (s)",
    "Reason",
  ];

  // CSV Rows
  const rows = batchResultsData.map((item) => {
    // Escape quotes and wrap in quotes to handle commas in story
    const story = `"${item.story.replace(/"/g, '""')}"`;
    const category = `"${item.category}"`;
    const reason = `"${item.reason.replace(/"/g, '""')}"`;
    return [story, item.classification, category, item.latency, reason].join(
      ",",
    );
  });

  const csvContent = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.setAttribute(
    "download",
    `batch_results_${new Date().toISOString().slice(0, 10)}.csv`,
  );
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}


// --- Comparison Logic ---
async function runModelComparison() {
  const storyInput = document.getElementById("compareStoryInput");
  const strategyInput = document.getElementById("compareStrategySelect");
  const checkboxes = document.querySelectorAll("#compareModelCheckboxes input[type='checkbox']:checked");
  const errorBanner = document.getElementById("compareErrorBanner");
  const errorText = document.getElementById("compareErrorText");
  const resultsArea = document.getElementById("compareResultsArea");
  const cardsContainer = document.getElementById("compareCardsContainer");
  const runBtn = document.getElementById("runCompareBtn");

  // Basic Validation
  if (!storyInput.value.trim()) {
    showCompareError("Please enter a user story to compare.");
    return;
  }
  if (!isValidUserStory(storyInput.value.trim())) {
    showCompareError("Please enter a valid user story (e.g., 'As a user, I want...').");
    return;
  }
  if (checkboxes.length < 2) {
    showCompareError("Minimum 2 models required to compare.");
    return;
  }

  // Clear errors and show loading state
  if (errorBanner) errorBanner.classList.add("hidden");
  resultsArea.classList.remove("hidden");
  cardsContainer.innerHTML = "";
  
  const selectedModels = Array.from(checkboxes).map(cb => ({
    value: cb.value,
    label: cb.nextElementSibling.textContent.trim()
  }));
  const strategy = strategyInput.value;
  const story = storyInput.value.trim();

  runBtn.disabled = true;
  runBtn.innerHTML = '<svg class="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Running...';

  // Create loading cards for each selected model
  selectedModels.forEach(model => {
    const card = document.createElement("div");
    card.id = `compare-card-${model.value}`;
    card.className = "glass-card rounded-2xl overflow-hidden bg-white/70 dark:bg-slate-800/70 opacity-50 transition-opacity duration-300 flex flex-col h-full";
    card.innerHTML = `
      <div class="px-6 py-4 border-b border-gray-200 dark:border-slate-700 bg-gray-50/50 dark:bg-slate-800/30 flex justify-between items-center">
        <h4 class="font-bold text-gray-900 dark:text-white flex items-center">
            <svg class="w-4 h-4 mr-2 text-blue-500 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            ${model.label}
        </h4>
      </div>
      <div class="p-6 flex-grow flex items-center justify-center">
        <p class="text-sm text-gray-400 animate-pulse">Waiting for response...</p>
      </div>
    `;
    cardsContainer.appendChild(card);
  });

  // Run requests concurrently
  const promises = selectedModels.map(async (model) => {
    try {
      const response = await fetch("/single", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ story, model: model.value, strategy })
      });
      
      const data = await response.json();
      const card = document.getElementById(`compare-card-${model.value}`);
      card.classList.remove("opacity-50");
      
      if (!response.ok) {
        card.innerHTML = `
          <div class="px-6 py-4 border-b border-gray-200 dark:border-slate-700 bg-red-50/50 dark:bg-red-900/10">
            <h4 class="font-bold text-gray-900 dark:text-white flex items-center">
                <span class="w-2 h-2 rounded-full bg-red-500 mr-2"></span>${model.label}
            </h4>
          </div>
          <div class="p-6 flex-grow flex flex-col justify-center">
            <p class="text-sm text-red-500">${data.error || "An error occurred."}</p>
          </div>
        `;
        return;
      }

      const isFR = data.classification === "FR";
      const badgeClass = isFR ? "text-green-600 bg-green-100 border-green-200 dark:text-green-400 dark:bg-green-900/30 dark:border-green-500/30" : "text-red-600 bg-red-100 border-red-200 dark:text-red-400 dark:bg-red-900/30 dark:border-red-500/30";
      
      card.innerHTML = `
        <div class="px-6 py-4 border-b border-gray-200 dark:border-slate-700 bg-gray-50/50 dark:bg-slate-800/30 flex justify-between items-center">
          <h4 class="font-bold text-gray-900 dark:text-white flex items-center">
              <span class="w-2 h-2 rounded-full bg-green-500 mr-2"></span>${model.label}
          </h4>
          <span class="text-xs font-mono text-gray-500">${data.latency}s</span>
        </div>
        <div class="p-6 flex-grow flex flex-col space-y-4">
          <div class="flex items-center space-x-3">
             <span class="px-2 py-1 rounded text-xs font-bold border ${badgeClass}">
                ${data.classification}
             </span>
             <span class="text-sm font-medium text-gray-700 dark:text-gray-300">${data.category || "-"}</span>
          </div>
          <div class="bg-gray-50 dark:bg-slate-900/50 p-3 rounded-lg border border-gray-200 dark:border-slate-700/50 flex-grow">
            <p class="text-xs text-gray-400 uppercase font-bold mb-1">Reason</p>
            <p class="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">${data.reason || "--"}</p>
          </div>
        </div>
      `;
    } catch (e) {
      const card = document.getElementById(`compare-card-${model.value}`);
      if(card) {
          card.classList.remove("opacity-50");
          card.innerHTML = `
            <div class="px-6 py-4 border-b border-gray-200 dark:border-slate-700 bg-red-50/50 dark:bg-red-900/10">
              <h4 class="font-bold text-gray-900 dark:text-white flex items-center">
                  <span class="w-2 h-2 rounded-full bg-red-500 mr-2"></span>${model.label}
              </h4>
            </div>
            <div class="p-6 flex-grow flex flex-col justify-center">
              <p class="text-sm text-red-500">Network error occurred.</p>
            </div>
          `;
      }
    }
  });

  await Promise.all(promises);

  // Restore button
  runBtn.disabled = false;
  runBtn.innerHTML = `
      <span>Run Comparison</span>
      <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path>
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
      </svg>
  `;

  function showCompareError(msg) {
    if (errorBanner && errorText) {
      errorText.textContent = msg;
      errorBanner.classList.remove("hidden");
    }
  }
}

function clearComparison() {
  const storyInput = document.getElementById("compareStoryInput");
  const errorBanner = document.getElementById("compareErrorBanner");
  const resultsArea = document.getElementById("compareResultsArea");
  const cardsContainer = document.getElementById("compareCardsContainer");
  const runBtn = document.getElementById("runCompareBtn");

  if(storyInput) storyInput.value = "";
  if(errorBanner) errorBanner.classList.add("hidden");
  if(resultsArea) resultsArea.classList.add("hidden");
  if(cardsContainer) cardsContainer.innerHTML = "";
  
  if(runBtn) {
    runBtn.disabled = false;
    runBtn.innerHTML = `
        <span>Run Comparison</span>
        <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path>
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
        </svg>
    `;
  }
}

// --- Analytics Charts --- 

// global variables for charts to allow updates
let frChart = null;
let categoryChart = null;
let latencyChart = null;
let techniqueChart = null;

async function loadRuns() {
  const res = await fetch("/api/batch_runs");
  const runs = await res.json();

  const selector = document.getElementById("runSelector");

  runs.forEach(run => {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent =
      `Run ${run.id} - ${run.model} (${run.prompting_technique})`;
    selector.appendChild(option);
  });
}

async function initCharts() {

  // Shared Chart Config
  Chart.defaults.color = "#94a3b8";
  Chart.defaults.borderColor = "#334155";

  const selectedRun = document.getElementById("runSelector").value;

  let url = "/api/analytics_data";
    if (selectedRun) {
    url += `?batch_run_id=${selectedRun}`;
  }

  const response = await fetch(url);
  const data = await response.json();

  const fr = data.fr || 0;
  const nfr = data.nfr || 0;
  const categories = data.categories || {};
  const latencies = data.latencies || [];
  const labels = Object.keys(categories);
  const values = Object.values(categories);

  document.getElementById("totalStories").innerText = data.total;
  document.getElementById("frCount").innerText = data.fr;
  document.getElementById("nfrCount").innerText = data.nfr;

  // Simple accuracy calculation for display
  let accuracy = 0;
  if(data.total > 0){
     accuracy = Math.round((data.fr + data.nfr)/data.total * 100);
  }
  
  document.getElementById("avgAccuracy").innerText = accuracy + "%";
  
  // Need to remove multiple potential listeners to prevent dupes, 
  // or just overwrite onchange. Given the structure, assigning onchange is safer.
  const selectorEl = document.getElementById("runSelector");
  if(selectorEl) {
      selectorEl.onchange = () => {
          initCharts();
          loadTechniqueComparison();
      };
  }
  
  //fallback for zero data
  if(data.total === 0){
   document.getElementById("totalStories").innerText = "0";
   document.getElementById("avgAccuracy").innerText = "--";
   document.getElementById("frCount").innerText = "0";
   document.getElementById("nfrCount").innerText = "0";
  }
  

    // ---------- DESTROY OLD CHARTS ----------
  // frChart is 'plotly' string when Plotly was used — only call destroy() on real Chart.js instances
  if (frChart && typeof frChart.destroy === 'function') frChart.destroy();
  if (categoryChart && typeof categoryChart.destroy === 'function') categoryChart.destroy();
  if (latencyChart && typeof latencyChart.destroy === 'function') latencyChart.destroy();
  frChart = null; categoryChart = null; latencyChart = null;
  // Purge any existing Plotly containers so they can be re-drawn
  const existingPlotly = document.getElementById('analyticsFrPlotly');
  if (existingPlotly && typeof Plotly !== 'undefined') Plotly.purge(existingPlotly);

  //debugging line for checking data output in console
  console.log("Analytics data:", data);

  // FR vs NFR (Sunburst using Plotly)
  // On first load ctx1 is the canvas; on subsequent loads the canvas is gone
  // (replaced by the Plotly div), so we fall back to the plotly container directly.
  const ctx1 = document.getElementById("frNfrChart");
  const plotlyContainer = document.getElementById("analyticsFrPlotly");

  if (ctx1) {
    // First load: replace canvas with a Plotly div
    const parent = ctx1.parentElement;
    parent.innerHTML = '<div id="analyticsFrPlotly" style="width:100%; height:100%;"></div>';
  } else if (!plotlyContainer) {
    // Safety: neither element found — nothing to draw into
    console.warn("No container found for FR/NFR sunburst chart");
  }

  // At this point analyticsFrPlotly always exists (either just created or from a previous render)
  const sunburstTarget = document.getElementById("analyticsFrPlotly");
  if (sunburstTarget) {
    const sunburstData = [{
      type: "sunburst",
      labels: ["Total", "Functional", "Non-Functional"],
      parents: ["", "Total", "Total"],
      values: [data.total, fr, nfr],
      branchvalues: "total",
      marker: {
        colors: ["#1e293b", "#3b82f6", "#ef4444"]
      },
      textinfo: "label+value+percent parent",
      hoverinfo: "label+value+percent root",
      insidetextfont: { family: "'Inter', sans-serif", size: 14, color: "#ffffff" },
    }];

    const layout = {
      margin: { l: 0, r: 0, b: 0, t: 0 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent"
    };

    if (data.total > 0) {
      Plotly.react('analyticsFrPlotly', sunburstData, layout, {responsive: true, displayModeBar: false});
    }
    frChart = { destroy: () => { Plotly.purge('analyticsFrPlotly'); } };
  }

  // NFR Categories (Premium Radar Chart)
  const ctx2 = document.getElementById("nfrCategoryChart");
  if (ctx2) {
    categoryChart = new Chart(ctx2, {
      type: "radar",
      data: {
        labels: labels.length ? labels : ["No Data"],
        datasets: [{
            label: "Category Frequency",
            data: values.length ? values : [0],
            backgroundColor: "rgba(139, 92, 246, 0.25)",
            borderColor: "#8b5cf6",
            pointBackgroundColor: "#c4b5fd",
            pointBorderColor: "#fff",
            pointHoverBackgroundColor: "#fff",
            pointHoverBorderColor: "#8b5cf6",
            borderWidth: 2,
            fill: true
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        elements: {
          line: { tension: 0.3 }
        },
        plugins: { 
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(15, 23, 42, 0.9)",
            titleColor: "#f8fafc",
            bodyColor: "#cbd5e1",
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1,
            padding: 10,
            displayColors: false
          }
        },
        scales: {
          r: {
            angleLines: { color: "rgba(255, 255, 255, 0.06)" },
            grid: { color: "rgba(255, 255, 255, 0.06)", circular: true },
            pointLabels: {
              color: "#94a3b8",
              font: { family: "'Inter', sans-serif", size: 10, weight: '500' }
            },
            ticks: {
              display: false,
              stepSize: 1,
              beginAtZero: true
            }
          }
        }
      }
    });
  }

  // Latency Line Chart
  const ctx3 = document.getElementById("latencyChart");
  if (ctx3 && latencies.length > 0) {
    latencyChart = new Chart(ctx3, {
      type: "line",
      data: {
        labels: latencies.map((_, i) => `Story ${i+1}`),
        datasets: [{
          label: "Latency (s)",
          data: latencies,
          borderColor: "#10b981",
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
      },
    });
  }
}

//Technique Chart
async function loadTechniqueComparison() {
  const selectedRun = document.getElementById("runSelector") ? document.getElementById("runSelector").value : "";
  let url = "/api/technique_comparison";
  if (selectedRun) {
    url += `?batch_run_id=${selectedRun}`;
  }
  const res = await fetch(url);
  const data = await res.json();

  const labels = data.map(d => d.prompting_technique);
  const frValues = data.map(d => d.fr);
  const nfrValues = data.map(d => d.nfr);

  const ctx = document.getElementById("techniqueComparisonChart");

  if (!ctx) return;

  // ✅ Destroy previous chart
  if (techniqueChart) {
    techniqueChart.destroy();
  }

  // Restructure the data for a Treemap
  // We want boxes for each Technique, and inside those boxes, FR and NFR
  const treemapData = [];
  for(let i=0; i<labels.length; i++) {
    treemapData.push({ technique: labels[i], type: "Functional (FR)", value: frValues[i] });
    treemapData.push({ technique: labels[i], type: "Non-Functional (NFR)", value: nfrValues[i] });
  }

  techniqueChart = new Chart(ctx, {
    type: "treemap",
    data: {
      datasets: [
        {
          label: 'Technique Comparison',
          tree: treemapData,
          key: "value",
          groups: ["technique", "type"],
          backgroundColor: (ctx) => {
            if (ctx.type !== 'data') return 'transparent';
            // Color based on FR vs NFR
            return ctx.raw._data.type === 'Functional (FR)' ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)';
          },
          borderColor: '#1e293b',
          borderWidth: 1,
          spacing: 1,
          fontColor: '#ffffff',
          labels: {
            display: true,
            align: 'center',
            position: 'middle',
            font: { family: "'Inter', sans-serif", size: 12, weight: 'bold' },
            color: '#ffffff',
            formatter: (ctx) => {
                if(ctx.type !== 'data') return [];
                return [ctx.raw._data.type, ctx.raw._data.value];
            }
          },
          captions: {
             display: true,
             align: 'center',
             color: '#e2e8f0',
             font: { family: "'Inter', sans-serif", size: 14, weight: 'bold' },
             padding: 4
          }
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              if(!items[0]) return '';
              const r = items[0].raw;
              // Tooltip logic depending on if hovering over group or leaf
              return r._data ? r._data.technique : r.g;
            },
            label: (item) => {
              const r = item.raw;
              if (r._data) {
                  return `${r._data.type}: ${r._data.value}`;
              } else {
                  return `Total: ${r.v}`;
              }
            }
          }
        }
      }
    }
  });
}
// Load charts and misc listeners after DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  loadTechniqueComparison();

  document.addEventListener("input", (e) => {
    if (e.target.id === "volumeSlider") {
      const valEl = document.getElementById("volumeValue");
      if (valEl) valEl.textContent = e.target.value;
    }
  });

  // ✅ SAFELY ATTACH BUTTON EVENTS
  const groupBatchBtn = document.getElementById("groupBatchBtn");
  const groupNormalBtn = document.getElementById("groupNormalBtn");

  if (groupBatchBtn) {
    groupBatchBtn.addEventListener("click", () => {
      currentMode = "similarity";
      processBatch();
    });
  }

  if (groupNormalBtn) {
    groupNormalBtn.addEventListener("click", () => {
      currentMode = "normal";
      processBatch();
    });
  }
});
// =========================
// REAL-TIME FEEDBACK SYSTEM
// =========================

function showCorrectionBox() {
  const box = document.getElementById("correctionBox");
  if (box) box.classList.remove("hidden");
}

async function sendFeedback(isCorrect) {

  if (!window.currentResultId) {
    alert("No result available.");
    return;
  }

  let correctedLabel = null;

if (!isCorrect) {
  correctedLabel = document.getElementById("correctLabel")?.value;

  if (!correctedLabel) {
    alert("Please select the correct label.");
    return;
  }
}

  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        result_id: window.currentResultId,
        is_correct: isCorrect,
        corrected_label: isCorrect ? null : correctedLabel,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      alert(data.error || "Feedback failed");
      return;
    }

    // ✅ Success UI
    alert("✅ Feedback saved!");

    // Optional: auto reclassify if corrected
    if (!isCorrect && data.updated_result) {
      updateResultUI(data.updated_result);
    }

  } catch (err) {
    console.error(err);
    alert("❌ Network error while sending feedback");
  }
}
function updateResultUI(data) {

  const badge = document.getElementById("resBadge");

  if (badge) {
    badge.textContent = data.classification;
    badge.className = `text-2xl font-bold px-4 py-2 rounded-lg border ${
      data.classification === "FR" ? "badge-fr" : "badge-nfr"
    }`;
  }

  document.getElementById("resCategory").textContent =
    data.category || "-";

  document.getElementById("resReason").innerText =
    data.reason || "--";

}