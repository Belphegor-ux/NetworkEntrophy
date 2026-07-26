let currentDataset = 'Karate';
let currentMode = 'Static';
let selectedMethods = new Set();
let mainChart = null;

const colors = [
    '#38bdf8', '#818cf8', '#f43f5e', '#10b981', '#fbbf24', 
    '#a855f7', '#ec4899', '#6366f1', '#14b8a6', '#f97316'
];

function initDashboard() {
    if (typeof chartData === 'undefined') {
        console.error('Data not loaded');
        document.getElementById('status').textContent = 'Error: Data not loaded';
        document.getElementById('status').style.color = '#f43f5e';
        return;
    }
    console.log('Data loaded:', chartData);

    // Populate the dataset dropdown dynamically from whatever networks are in the data.
    populateDatasetSelect();

    // Populate methods for the first dataset
    updateMethodList();
    
    // Dataset change listener
    document.getElementById('dataset-select').addEventListener('change', (e) => {
        currentDataset = e.target.value;
        selectedMethods.clear();
        updateMethodList();
        updateChart();
    });

    // Mode change listener
    document.getElementById('mode-select').addEventListener('change', (e) => {
        currentMode = e.target.value;
        selectedMethods.clear();
        updateMethodList();
        updateChart();
    });

    document.getElementById('reset-btn').addEventListener('click', () => {
        selectedMethods.clear();
        updateMethodList();
        updateChart();
    });

    updateChart();
    document.getElementById('status').textContent = 'Live Analysis Ready';
}

function populateDatasetSelect() {
    const sel = document.getElementById('dataset-select');
    const keys = Object.keys(chartData);
    if (keys.length === 0) return;
    sel.innerHTML = '';
    keys.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
    });
    // Keep current selection valid.
    if (!keys.includes(currentDataset)) {
        currentDataset = keys[0];
    }
    sel.value = currentDataset;
}

function updateMethodList() {
    const methodList = document.getElementById('method-list');
    methodList.innerHTML = '';
    
    if (typeof chartData === 'undefined' || !chartData[currentDataset]) return;
    
    let methods = [];
    if (currentMode === 'Comparative') {
        const staticMethods = Object.keys(chartData[currentDataset]['Static'] || {});
        const iterativeMethods = Object.keys(chartData[currentDataset]['Iterative'] || {});
        methods = [...new Set([...staticMethods, ...iterativeMethods])];
    } else {
        if (!chartData[currentDataset][currentMode]) return;
        methods = Object.keys(chartData[currentDataset][currentMode]);
    }
    
    methods.forEach((method, index) => {
        const item = document.createElement('div');
        item.className = `method-item ${selectedMethods.has(method) ? 'active' : ''}`;
        item.style.borderLeft = `4px solid ${colors[index % colors.length]}`;
        
        item.innerHTML = `
            <div class="checkbox-custom"></div>
            <span style="font-size: 0.875rem;">${method}</span>
        `;
        
        item.addEventListener('click', () => {
            if (selectedMethods.has(method)) {
                selectedMethods.delete(method);
            } else {
                selectedMethods.add(method);
            }
            item.classList.toggle('active');
            updateChart();
        });
        
        methodList.appendChild(item);
    });
}

function updateChart() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    
    if (mainChart) {
        mainChart.destroy();
    }
    
    const datasets = [];
    const statsGrid = document.getElementById('stats-grid');
    statsGrid.innerHTML = '';
    
    if (typeof chartData === 'undefined' || !chartData[currentDataset]) return;

    let methodsToGraph = selectedMethods.size > 0 
        ? Array.from(selectedMethods) 
        : [];
    
    if (methodsToGraph.length === 0) {
        if (currentMode === 'Comparative') {
            const staticMethods = Object.keys(chartData[currentDataset]['Static'] || {});
            const iterativeMethods = Object.keys(chartData[currentDataset]['Iterative'] || {});
            methodsToGraph = [...new Set([...staticMethods, ...iterativeMethods])];
        } else if (chartData[currentDataset][currentMode]) {
            methodsToGraph = Object.keys(chartData[currentDataset][currentMode]);
        }
    }

    let idx = 0;
    methodsToGraph.forEach(method => {
        if (currentMode === 'Comparative') {
            const staticData = chartData[currentDataset]['Static'] ? chartData[currentDataset]['Static'][method] : null;
            const iterativeData = chartData[currentDataset]['Iterative'] ? chartData[currentDataset]['Iterative'][method] : null;

            if (staticData) {
                datasets.push({
                    label: `${method} (Static)`,
                    data: staticData.x.map((x, i) => ({ x: x, y: staticData.y[i] })),
                    borderColor: colors[idx % colors.length],
                    borderDash: [5, 5],
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false
                });
            }

            if (iterativeData) {
                datasets.push({
                    label: `${method} (Iterative)`,
                    data: iterativeData.x.map((x, i) => ({ x: x, y: iterativeData.y[i] })),
                    borderColor: colors[idx % colors.length],
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false
                });
            }

            if (staticData || iterativeData) {
                const card = document.createElement('div');
                card.className = 'stat-card';
                let statHtml = `<div class="stat-label">${method}</div>`;
                if (staticData) statHtml += `<div class="stat-value" style="color: ${colors[idx % colors.length]}; opacity: 0.7; font-size: 1rem;">S: ${staticData.auc.toFixed(4)}</div>`;
                if (iterativeData) statHtml += `<div class="stat-value" style="color: ${colors[idx % colors.length]}">I: ${iterativeData.auc.toFixed(4)}</div>`;
                card.innerHTML = statHtml;
                statsGrid.appendChild(card);
            }
        } else {
            const data = chartData[currentDataset][currentMode] ? chartData[currentDataset][currentMode][method] : null;
            if (!data) return;

            datasets.push({
                label: method,
                data: data.x.map((x, i) => ({ x: x, y: data.y[i] })),
                borderColor: colors[idx % colors.length],
                backgroundColor: colors[idx % colors.length] + '20',
                borderWidth: 3,
                pointRadius: 0,
                pointHoverRadius: 5,
                tension: 0.1,
                fill: false
            });

            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `
                <div class="stat-label">${method}</div>
                <div class="stat-value" style="color: ${colors[idx % colors.length]}">${data.auc.toFixed(4)}</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary)">AUC Score</div>
            `;
            statsGrid.appendChild(card);
        }
        idx++;
    });

    mainChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Fraction of Edges Removed', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' },
                    min: 0,
                    max: 1
                },
                y: {
                    title: { display: true, text: 'Relative Giant Component Size', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    display: currentMode === 'Comparative',
                    labels: { color: '#94a3b8', boxWidth: 20, usePointStyle: false }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: '#1e293b',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            }
        }
    });

    document.getElementById('chart-title').textContent = `${currentDataset} Dismantling Analysis (${currentMode})`;
}

window.onload = initDashboard;
