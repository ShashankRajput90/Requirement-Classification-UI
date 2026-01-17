// dashboard.js

// --- Global Config ---
// In a real app, these would come from the sidebar state
function getSelectedModel() {
    return document.getElementById('modelSelect') ? document.getElementById('modelSelect').value : 'ChatGPT';
}

function getSelectedStrategy() {
    return document.getElementById('strategySelect') ? document.getElementById('strategySelect').value : 'Zero-shot';
}

// --- Single Classification ---
async function classifyStory() {
    const input = document.getElementById('storyInput');
    const resultArea = document.getElementById('resultArea');
    const indicator = document.getElementById('statusIndicator');

    if (!input || !input.value.trim()) {
        alert("Please enter a user story.");
        return;
    }

    // UI Updates
    if (indicator) indicator.classList.remove('hidden');
    if (resultArea) {
        resultArea.classList.add('hidden');
        resultArea.classList.remove('opacity-100');
        resultArea.classList.add('opacity-0');
    }

    try {
        const response = await fetch('/single', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                story: input.value,
                model: getSelectedModel(),
                strategy: getSelectedStrategy()
            })
        });
        const data = await response.json();

        // Update Result Cards
        const badge = document.getElementById('resBadge');
        if (badge) {
            badge.textContent = data.classification;
            badge.className = `text-2xl font-bold px-4 py-2 rounded-lg border ${data.classification === 'FR' ? 'badge-fr' : 'badge-nfr'}`;
        }

        document.getElementById('resCategory').textContent = data.category || 'Functional Requirement';
        document.getElementById('resModel').textContent = data.model;
        document.getElementById('resStrategy').textContent = data.strategy;
        document.getElementById('resLatency').textContent = `${data.latency}s`;

        // Circle Animation removed

        // Show Results
        if (resultArea) {
            resultArea.classList.remove('hidden');
            // Small delay to allow display:block to apply before opacity transition
            setTimeout(() => {
                resultArea.classList.remove('opacity-0');
                resultArea.classList.add('opacity-100');
            }, 50);
        }

    } catch (e) {
        console.error(e);
        alert("Error classifying story.");
    } finally {
        if (indicator) indicator.classList.add('hidden');
    }
}

// --- File Upload Logic ---
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadPrompt = document.getElementById('uploadPrompt');
const fileActionBar = document.getElementById('fileActionBar');
const uploadedFilename = document.getElementById('uploadedFilename');
const removeBtn = document.getElementById('removeBtn');
const previewBtn = document.getElementById('previewBtn');

if (dropZone && fileInput) {
    // Only trigger file picker if clicking on dropZone BUT NOT if clicking actions
    dropZone.addEventListener('click', (e) => {
        if (!e.target.closest('button')) {
            fileInput.click();
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-blue-500', 'bg-slate-800/10');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-blue-500', 'bg-slate-800/10');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-blue-500', 'bg-slate-800/10');

        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Stop bubble to dropZone
            removeFile();
        });
    }

    if (previewBtn) {
        previewBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            previewFile();
        });
    }
}

function handleFileSelect(file) {
    if (uploadedFilename) uploadedFilename.textContent = file.name;
    if (uploadPrompt) uploadPrompt.classList.add('opacity-0', 'pointer-events-none'); // Hide prompt
    if (fileActionBar) fileActionBar.classList.remove('hidden');
}

function removeFile() {
    fileInput.value = ''; // Clear input
    if (uploadPrompt) uploadPrompt.classList.remove('opacity-0', 'pointer-events-none'); // Show prompt
    if (fileActionBar) fileActionBar.classList.add('hidden'); // Hide actions
}

function previewFile() {
    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        const text = e.target.result;
        const rows = text.split('\n').slice(0, 101); // Top 100 rows + header

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
                } else if (line[i] === ',' && !inQuotes) {
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
        const bodyRows = rows.slice(1).filter(r => r.trim());

        const headerRowEl = document.getElementById('previewHeaderRow');
        const bodyEl = document.getElementById('previewBody');

        if (headerRowEl) {
            headerRowEl.innerHTML = header.map(h => `<th class="px-6 py-3 bg-slate-800 sticky top-0 z-10 font-bold text-gray-200 border-b border-slate-700 shadow-sm text-xs uppercase tracking-wider whitespace-nowrap">${h}</th>`).join('');
        }
        if (bodyEl) {
            bodyEl.innerHTML = bodyRows.map(row => {
                const cells = parseCSVLine(row);
                return `<tr class="border-b border-slate-700/50 hover:bg-slate-800/50 transition-colors">${cells.map(c => `<td class="px-6 py-3 whitespace-nowrap">${c}</td>`).join('')}</tr>`
            }).join('');
        }

        document.getElementById('previewModal').classList.remove('hidden');
    }
    reader.readAsText(file);
}

function closePreview() {
    document.getElementById('previewModal').classList.add('hidden');
}

// --- Batch Processing ---
let batchFrChart = null;
let batchCatChart = null;
let batchResultsData = []; // Store results for export

function setupBatchCharts() {
    // ... (Chart setup code remains same, omitted for brevity if possible, but replace tool needs context. 
    // Since I'm replacing a large chunk or just adding variables, let's keep it safe.)
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#334155';

    // FR vs NFR
    const ctx1 = document.getElementById('batchFrChart');
    if (ctx1) {
        if (batchFrChart) batchFrChart.destroy();
        batchFrChart = new Chart(ctx1, {
            type: 'doughnut',
            data: {
                labels: ['Functional', 'Non-Functional'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { 
                    legend: { position: 'right' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                let value = context.parsed;
                                let total = context.chart._metasets[context.datasetIndex].total;
                                let percentage = Math.round((value / total) * 100) + '%';
                                return label + value + ' (' + percentage + ')';
                            }
                        }
                    }
                }
            }
        });
    }

    // Categories
    const ctx2 = document.getElementById('batchCatChart');
    if (ctx2) {
        if (batchCatChart) batchCatChart.destroy();
        batchCatChart = new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: ["Accuracy", "Usability", "Performance", "Efficiency", "Security", "Privacy", "Fairness & Bias", "Explainability", "Interpretability", "Transparency", "Accessibility", "Reliability", "Robustness", "Maintainability", "Scalability", "Interoperability", "Completeness & Consistency", "Trust", "Safety & Governance"],
                datasets: [{
                    label: 'Count',
                    data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    backgroundColor: '#3b82f6',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { 
                    y: { 
                        beginAtZero: true, 
                        grid: { color: '#334155' },
                        ticks: { stepSize: 1, precision: 0 }
                    } 
                }
            }
        });
    }
}

async function processBatch() {
    const tableBody = document.getElementById('resultsTableBody');
    const batchResults = document.getElementById('batchResults');
    const volume = document.getElementById('volumeSlider').value;
    const fileInput = document.getElementById('fileInput');
    const exportBtn = document.getElementById('exportBtn');

    if (tableBody) tableBody.innerHTML = '<tr><td colspan="4" class="px-6 py-4 text-center text-gray-500">Processing...</td></tr>';
    if (batchResults) batchResults.classList.remove('hidden');
    
    // Reset data
    batchResultsData = [];
    if (exportBtn) exportBtn.disabled = true;

    setupBatchCharts();

    const formData = new FormData();
    formData.append('count', volume);
    formData.append('model', getSelectedModel());
    formData.append('strategy', getSelectedStrategy());

    if (fileInput && fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
    }

    try {
        const response = await fetch('/batch', {
            method: 'POST',
            body: formData
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
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');

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

function handleStreamData(data) {
    if (data.type === 'result') {
        // Collect data for export
        batchResultsData.push({
            story: data.story,
            classification: data.result.classification,
            category: data.result.category || '',
            latency: data.result.latency,
            reason: data.result.reason || ''
        });

        // Enable export button if we have data
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn && batchResultsData.length > 0) {
            exportBtn.disabled = false;
        }

        // Update Stats
        const stats = data.current_stats;
        document.getElementById('totalCount').textContent = stats.total;
        document.getElementById('frCount').textContent = stats.fr_count;
        document.getElementById('nfrCount').textContent = stats.nfr_count;
        document.getElementById('avgTime').textContent = `${stats.avg_time}s`;

        // Update Charts
        if (batchFrChart) {
            batchFrChart.data.datasets[0].data = [stats.fr_count, stats.nfr_count];
            batchFrChart.update('none');
        }
        if (batchCatChart && stats.category_counts) {
            const labels = batchCatChart.data.labels;
            const newData = labels.map(l => stats.category_counts[l] || 0);
            batchCatChart.data.datasets[0].data = newData;
            batchCatChart.update('none');
        }

        // Add Row
        const tableBody = document.getElementById('resultsTableBody');
        if (tableBody) {
            // Remove "Processing..." loading row if it exists
            if (tableBody.querySelector('td[colspan="4"]')) {
                tableBody.innerHTML = '';
            }

            const item = data;
            const row = document.createElement('tr');
            row.className = "fade-in"; // Add animation
            const isFR = item.result.classification === 'FR';
            row.innerHTML = `
                <td class="px-6 py-4 font-medium text-gray-900 dark:text-white">${item.story}</td>
                <td class="px-6 py-4">
                    <span class="${isFR ? 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30' : 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30'} px-2 py-1 rounded text-xs font-bold border ${isFR ? 'border-green-200 dark:border-green-500/30' : 'border-red-200 dark:border-red-500/30'}">
                        ${item.result.classification}
                    </span>
                </td>
                <td class="px-6 py-4 text-gray-500 dark:text-gray-400">${item.result.category || '-'}</td>
                <td class="px-6 py-4 text-gray-500 dark:text-gray-400 text-sm">${item.result.reason || '-'}</td>
                <td class="px-6 py-4 font-mono text-gray-500">${item.result.latency}s</td>
            `;
            // Prepend or append? Append is more natural for stream
            tableBody.appendChild(row);
        }
    }
}

function exportBatchCSV() {
    if (!batchResultsData || batchResultsData.length === 0) return;

    // CSV Header
    const headers = ['User Story', 'Classification', 'Category', 'Latency (s)', 'Reason'];
    
    // CSV Rows
    const rows = batchResultsData.map(item => {
        // Escape quotes and wrap in quotes to handle commas in story
        const story = `"${item.story.replace(/"/g, '""')}"`;
        const category = `"${item.category}"`;
        const reason = `"${item.reason.replace(/"/g, '""')}"`;
        return [story, item.classification, category, item.latency, reason].join(',');
    });

    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `batch_results_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// --- Comparison Logic ---
async function loadComparisonData() {
    const tableBody = document.getElementById('comparisonTableBody');
    if (!tableBody) return;

    tableBody.innerHTML = '<tr><td colspan="6" class="px-6 py-4 text-center">Loading benchmark data...</td></tr>';

    try {
        const response = await fetch('/api/comparison-data');
        const data = await response.json();

        tableBody.innerHTML = '';

        let bestModel = data[0]; // Assumes sorted by backend

        data.forEach((row, index) => {
            const tr = document.createElement('tr');
            if (index === 0) tr.className = "bg-green-500/10 border-l-4 border-green-500";

            tr.innerHTML = `
                <td class="px-6 py-4 font-medium text-white flex items-center">
                    ${row.model}
                    ${index === 0 ? '<svg class="w-4 h-4 ml-2 text-green-400" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path></svg>' : ''}
                </td>
                <td class="px-6 py-4">${Math.round(row.accuracy * 100)}%</td>
                <td class="px-6 py-4">${row.precision}</td>
                <td class="px-6 py-4">${row.recall}</td>
                <td class="px-6 py-4 font-bold text-blue-300">${row.f1}</td>
                <td class="px-6 py-4 font-mono text-gray-500">${row.avg_latency}s</td>
            `;
            tableBody.appendChild(tr);
        });

        const recText = document.getElementById('recommendationText');
        if (recText) {
            recText.innerHTML = `Based on the latest run, <strong class="text-white">${bestModel.model}</strong> is the top performer with an F1-score of <strong>${bestModel.f1}</strong>. It is recommended for production use cases requiring high accuracy.`;
        }

    } catch (e) {
        console.error(e);
    }
}

// --- Analytics Charts ---
function initCharts() {
    // Shared Chart Config
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#334155';

    // FR vs NFR
    const ctx1 = document.getElementById('frNfrChart');
    if (ctx1) {
        new Chart(ctx1, {
            type: 'doughnut',
            data: {
                labels: ['Functional', 'Non-Functional'],
                datasets: [{
                    data: [856, 392],
                    backgroundColor: ['#3b82f6', '#ef4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                let value = context.parsed;
                                let total = context.chart._metasets[context.datasetIndex].total;
                                let percentage = Math.round((value / total) * 100) + '%';
                                return label + value + ' (' + percentage + ')';
                            }
                        }
                    }
                }
            }
        });
    }

    // NFR Categories
    const ctx2 = document.getElementById('nfrCategoryChart');
    if (ctx2) {
        new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: ['Security', 'Performance', 'Usability', 'Reliability', 'Maintainability'],
                datasets: [{
                    label: 'Count',
                    data: [120, 80, 95, 60, 37],
                    backgroundColor: '#8b5cf6',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { 
                    y: { 
                        beginAtZero: true,
                        ticks: { stepSize: 1, precision: 0 }
                    } 
                }
            }
        });
    }

    // Latency Line Chart
    const ctx3 = document.getElementById('latencyChart');
    if (ctx3) {
        new Chart(ctx3, {
            type: 'line',
            data: {
                labels: ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5'],
                datasets: [
                    {
                        label: 'ChatGPT',
                        data: [1.2, 1.4, 1.1, 1.5, 1.3],
                        borderColor: '#10b981',
                        tension: 0.4
                    },
                    {
                        label: 'Gemini',
                        data: [0.8, 0.9, 0.85, 0.8, 0.9],
                        borderColor: '#3b82f6',
                        tension: 0.4
                    },
                    {
                        label: 'Claude',
                        data: [1.8, 2.0, 1.9, 1.7, 2.1],
                        borderColor: '#f59e0b',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } }
            }
        });
    }
}

// Misc: Update slider value
document.addEventListener('input', (e) => {
    if (e.target.id === 'volumeSlider') {
        document.getElementById('volumeValue').textContent = e.target.value;
    }
});
