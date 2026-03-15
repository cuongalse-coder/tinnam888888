/* ============================================
   TinNam AI - Frontend JavaScript
   ============================================ */

// ---- State ----
let currentTab = 'mega';
let charts = {};

// ---- MASTER PREDICTION ----
function runMaster(type) {
    const btn = document.getElementById('btn-master-' + type);
    const origText = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '⏳ AI đang phân tích...'; btn.style.animation = 'pulse 1s infinite';
    const container = document.getElementById('master-' + type);
    container.innerHTML = `<div class="card" style="border-color:#f43f5e;box-shadow:0 0 40px rgba(244,63,94,0.3);"><div class="loading">
        <div class="spinner" style="border-top-color:#f43f5e;"></div>
        <span class="loading-text">🎯 Đang tối ưu hóa 15 tín hiệu từ 70+ thuật toán...<br>Auto-tuning weights qua backtest. Vui lòng chờ 1-3 phút.</span></div></div>`;
    fetch(`/api/master/${type}`, {method:'POST',headers:{'Content-Type':'application/json'}})
    .then(r => r.json()).then(data => {
        btn.disabled = false; btn.innerHTML = origText; btn.style.animation = 'pulse-glow 2s infinite';
        if (data.error) { container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`; return; }
        renderMaster(container, data, type); showToast('Dự đoán hoàn tất! 🎯', 'success');
    }).catch(err => { btn.disabled=false;btn.innerHTML=origText;btn.style.animation='pulse-glow 2s infinite';
        container.innerHTML=`<div class="card"><div class="empty-state"><div class="icon">❌</div><p>${err.message}</p></div></div>`; });
}
function renderMaster(container, data, type) {
    const bt = data.backtest || {};
    const conf = data.confidence || {};
    const confColor = conf.level==='high'?'#22c55e':conf.level==='medium'?'#f59e0b':'#ef4444';
    const typeName = type === 'mega' ? 'Mega 6/45' : 'Power 6/55';
    let html = '';
    // Numbers (BIG)
    html += `<div class="card" style="border-color:#22c55e;box-shadow:0 0 50px rgba(34,197,94,0.3);background:linear-gradient(135deg,rgba(34,197,94,0.04),rgba(244,63,94,0.04));">
        <div style="text-align:center;">
            <div style="font-size:0.9rem;color:var(--text-muted);margin-bottom:8px;">${typeName} — Kỳ tiếp theo</div>
            <div style="font-size:1.1rem;font-weight:700;color:#22c55e;margin-bottom:16px;">🎯 DỰ ĐOÁN CHÍNH XÁC</div>
            <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin:16px 0 24px;">
                ${data.numbers ? data.numbers.map(n => `<span class="ball" style="background:linear-gradient(135deg,#f43f5e,#7c3aed);width:64px;height:64px;font-size:1.5rem;line-height:64px;box-shadow:0 6px 20px rgba(244,63,94,0.4);">${n.toString().padStart(2,'0')}</span>`).join('') : ''}
            </div>
            <div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;margin-top:16px;">
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:900;color:${confColor};">${conf.score||0}%</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);">Confidence</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:900;color:#6366f1;">${bt.avg||0}/6</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);">TB Backtest</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:900;color:#f59e0b;">${bt.max||0}/6</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);">Max</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:900;color:${(bt.improvement||0)>0?'#22c55e':'#ef4444'};">${(bt.improvement||0)>0?'+':''}${bt.improvement||0}%</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);">vs Random</div>
                </div>
            </div>
        </div></div>`;
    // Backtest detail
    html += `<div class="card">
        <div class="card-header"><div class="card-title"><span class="icon">🧪</span> Kết Quả Backtest (${bt.tests||0} kỳ)</div></div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;">
            <div style="text-align:center;padding:12px;background:var(--bg-glass);border-radius:10px;">
                <div style="font-size:1.3rem;font-weight:800;color:#22c55e;">${bt.match_3plus||0}</div>
                <div style="font-size:0.7rem;color:var(--text-muted);">Trúng 3+ số</div>
            </div>
            <div style="text-align:center;padding:12px;background:var(--bg-glass);border-radius:10px;">
                <div style="font-size:1.3rem;font-weight:800;color:#f59e0b;">${bt.match_4plus||0}</div>
                <div style="font-size:0.7rem;color:var(--text-muted);">Trúng 4+ số</div>
            </div>
            <div style="text-align:center;padding:12px;background:var(--bg-glass);border-radius:10px;">
                <div style="font-size:1.3rem;font-weight:800;color:#f43f5e;">${bt.match_5plus||0}</div>
                <div style="font-size:0.7rem;color:var(--text-muted);">Trúng 5+ số</div>
            </div>
            <div style="text-align:center;padding:12px;background:var(--bg-glass);border-radius:10px;">
                <div style="font-size:1.3rem;font-weight:800;color:#6366f1;">${bt.random_expected||0}</div>
                <div style="font-size:0.7rem;color:var(--text-muted);">Random TB</div>
            </div>
        </div>`;
    if (bt.distribution) {
        html += '<div style="margin-top:16px;"><strong>Phân bố số trùng:</strong><div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;">';
        for (const [k,v] of Object.entries(bt.distribution)) {
            const pct = (v / bt.tests * 100).toFixed(1);
            html += `<div style="padding:6px 12px;background:var(--bg-glass);border-radius:8px;text-align:center;">
                <div style="font-weight:700;">${k} số</div><div style="font-size:0.7rem;color:var(--text-muted);">${v}x (${pct}%)</div></div>`;
        }
        html += '</div></div>';
    }
    html += '</div>';
    // Score distribution
    if (data.score_details) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📊</span> Điểm Tin Cậy (Top 15 số)</div></div>`;
        data.score_details.forEach(s => {
            const isSelected = s.selected;
            const bg = isSelected ? 'background:linear-gradient(135deg,#f43f5e,#7c3aed);box-shadow:0 2px 10px rgba(244,63,94,0.3);' : '';
            html += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span class="ball small" style="${bg}">${s.number.toString().padStart(2,'0')}</span>
                <div style="flex:1;height:10px;background:var(--bg-tertiary);border-radius:5px;overflow:hidden;">
                    <div style="height:100%;width:${s.confidence}%;background:${isSelected?'linear-gradient(90deg,#f43f5e,#7c3aed)':'linear-gradient(90deg,#6366f1,#0ea5e9)'};border-radius:5px;transition:width 0.5s;"></div></div>
                <div style="font-size:0.75rem;min-width:55px;text-align:right;color:var(--text-muted);font-weight:${isSelected?'700':'400'};">${s.score} pts</div></div>`;
        });
        html += '</div>';
    }
    html += `<div style="text-align:center;font-size:0.75rem;color:var(--text-muted);margin-top:8px;">${data.method||''}</div>`;
    container.innerHTML = html;
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    loadData('mega', 20);
});

// ---- Tab Switching ----
function switchTab(type) {
    currentTab = type;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-' + type).classList.add('active');
    document.getElementById('content-' + type).classList.add('active');
    
    // Load data if table is empty
    const table = document.getElementById('table-' + type);
    if (table.querySelector('.loading') || table.querySelector('.empty-state')) {
        loadData(type, 20);
    }
}

// ---- Load Historical Data ----
function loadData(type, limit, btnEl) {
    // Update toggle button state
    if (btnEl) {
        btnEl.closest('.toggle-group').querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btnEl.classList.add('active');
    }
    
    const container = document.getElementById('table-' + type);
    container.innerHTML = `<div class="loading"><div class="spinner"></div><span class="loading-text">Đang tải dữ liệu...</span></div>`;
    
    fetch(`/api/data/${type}?limit=${limit}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = `<div class="empty-state"><div class="icon">📭</div><p>${data.error}</p><button class="btn btn-primary" onclick="scrapeData()">🔄 Thu Thập Dữ Liệu</button></div>`;
                return;
            }
            if (!data.data || data.data.length === 0) {
                container.innerHTML = `<div class="empty-state"><div class="icon">📭</div><p>Chưa có dữ liệu. Hãy thu thập dữ liệu trước!</p><button class="btn btn-primary" onclick="scrapeData()">🔄 Thu Thập Dữ Liệu</button></div>`;
                return;
            }
            renderTable(container, data.data, type);
        })
        .catch(err => {
            container.innerHTML = `<div class="empty-state"><div class="icon">❌</div><p>Lỗi tải dữ liệu: ${err.message}</p></div>`;
        });
}

function renderTable(container, rows, type) {
    const isPower = type === 'power';
    let html = `<div class="data-table-wrap"><table class="data-table">
        <thead><tr>
            <th>#</th>
            <th>Ngày</th>
            <th>Kết Quả</th>
            ${isPower ? '<th>Số ĐB</th>' : ''}
            <th>Jackpot</th>
        </tr></thead><tbody>`;
    
    rows.forEach((row, idx) => {
        const numbers = [row.n1, row.n2, row.n3, row.n4, row.n5, row.n6];
        const ballsHtml = numbers.map(n => `<span class="ball small">${n.toString().padStart(2, '0')}</span>`).join('');
        const bonusHtml = isPower ? `<td><span class="ball small bonus">${(row.bonus || 0).toString().padStart(2, '0')}</span></td>` : '';
        
        html += `<tr>
            <td style="color:var(--text-muted)">${idx + 1}</td>
            <td class="date-col">${row.draw_date}</td>
            <td><div class="ball-container">${ballsHtml}</div></td>
            ${bonusHtml}
            <td class="jackpot-col">${row.jackpot || '-'}</td>
        </tr>`;
    });
    
    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// ---- Prediction ----
function predict(type, deep = false) {
    const btnId = deep ? `btn-deep-${type}` : `btn-predict-${type}`;
    const btn = document.getElementById(btnId);
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = deep ? '⏳ Đang huấn luyện AI...' : '⏳ Đang dự đoán...';
    btn.classList.add('pulsing');
    
    const container = document.getElementById('prediction-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card"><div class="loading"><div class="spinner"></div><span class="loading-text">${deep ? 'Đang huấn luyện mô hình LSTM & Transformer... (có thể mất 1-2 phút)' : 'Đang phân tích và dự đoán...'}</span></div></div>`;
    
    fetch(`/api/predict/${type}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ train_deep: deep, n_sets: 3 })
    })
    .then(r => r.json())
    .then(data => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        btn.classList.remove('pulsing');
        
        if (data.error) {
            container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`;
            return;
        }
        renderPredictions(container, data, type);
        showToast('Dự đoán hoàn tất! 🎯', 'success');
    })
    .catch(err => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        btn.classList.remove('pulsing');
        container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">❌</div><p>Lỗi: ${err.message}</p></div></div>`;
        showToast('Lỗi dự đoán: ' + err.message, 'error');
    });
}

function renderPredictions(container, data, type) {
    const isPower = type === 'power';
    let html = '<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🔮</span> Kết Quả Dự Đoán</div>';
    html += `<span style="font-size:0.85rem; color:var(--text-muted)">Dựa trên ${data.total_draws} kỳ quay</span></div>`;
    
    // Render each model's predictions
    const models = data.models;
    for (const [key, model] of Object.entries(models)) {
        const statusClass = model.status === 'trained' ? 'trained' : 'fallback';
        const statusText = model.status === 'trained' ? '✅ Trained' : '⚡ Fast Mode';
        
        html += `<div class="prediction-card ${key === 'ensemble' ? 'style="border-color: var(--accent-4); background: rgba(245, 158, 11, 0.05);"' : ''}">`;
        html += `<div class="model-name">${model.icon} ${model.name} <span class="model-status">${statusText}</span></div>`;
        
        model.predictions.forEach((pred, idx) => {
            const balls = pred.map(n => `<span class="ball small">${n.toString().padStart(2, '0')}</span>`).join('');
            html += `<div class="prediction-set"><span class="set-label">Bộ ${idx + 1}:</span><div class="ball-container">${balls}</div></div>`;
        });
        
        html += '</div>';
    }
    
    html += '</div>';
    container.innerHTML = html;
}

// ---- Statistics ----
function loadStats(type) {
    const container = document.getElementById('stats-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card"><div class="loading"><div class="spinner"></div><span class="loading-text">Đang phân tích dữ liệu...</span></div></div>`;
    
    fetch(`/api/stats/${type}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`;
                return;
            }
            renderStats(container, data, type);
        })
        .catch(err => {
            container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">❌</div><p>Lỗi: ${err.message}</p></div></div>`;
        });
}

function renderStats(container, data, type) {
    const a = data.analysis;
    const maxNum = type === 'mega' ? 45 : 55;
    
    let html = '';
    
    // Summary stats
    html += `<div class="grid-4" style="margin-bottom:24px">
        <div class="stat-card"><div class="stat-value">${a.total_draws}</div><div class="stat-label">Tổng Số Kỳ</div></div>
        <div class="stat-card"><div class="stat-value">${a.avg_sum}</div><div class="stat-label">Tổng TB</div></div>
        <div class="stat-card"><div class="stat-value">${a.avg_odd_count}</div><div class="stat-label">TB Số Lẻ</div></div>
        <div class="stat-card"><div class="stat-value">${a.sum_range[0]}-${a.sum_range[1]}</div><div class="stat-label">Range Tổng</div></div>
    </div>`;
    
    // Hot & Cold numbers
    html += '<div class="grid-2">';
    
    // Hot Numbers
    html += '<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🔥</span> Số Nóng (Xuất hiện nhiều)</div></div>';
    a.hot_numbers.forEach(h => {
        const pct = (h.count / a.total_draws * 100).toFixed(1);
        html += `<div class="freq-bar-wrap">
            <span class="freq-number">${h.number.toString().padStart(2, '0')}</span>
            <div class="freq-bar-bg"><div class="freq-bar hot" style="width:${Math.min(pct * 4, 100)}%"></div></div>
            <span class="freq-count">${h.count}x</span>
        </div>`;
    });
    html += '</div>';
    
    // Cold Numbers
    html += '<div class="card"><div class="card-header"><div class="card-title"><span class="icon">❄️</span> Số Lạnh (Xuất hiện ít)</div></div>';
    a.cold_numbers.forEach(c => {
        const pct = (c.count / a.total_draws * 100).toFixed(1);
        html += `<div class="freq-bar-wrap">
            <span class="freq-number">${c.number.toString().padStart(2, '0')}</span>
            <div class="freq-bar-bg"><div class="freq-bar cold" style="width:${Math.min(pct * 4, 100)}%"></div></div>
            <span class="freq-count">${c.count}x</span>
        </div>`;
    });
    html += '</div></div>';
    
    // Overdue numbers
    html += '<div class="card"><div class="card-header"><div class="card-title"><span class="icon">⏰</span> Số Quá Hạn (Lâu chưa xuất hiện)</div></div>';
    html += '<div class="ball-container">';
    a.overdue_numbers.forEach(o => {
        html += `<div style="text-align:center; margin:8px;">
            <span class="ball">${o.number.toString().padStart(2, '0')}</span>
            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">${o.gap} kỳ</div>
        </div>`;
    });
    html += '</div></div>';
    
    // Recent Hot Numbers
    html += '<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📈</span> Xu Hướng Gần Đây (30 kỳ)</div></div>';
    html += '<div class="ball-container">';
    a.recent_hot.forEach(r => {
        html += `<div style="text-align:center; margin:8px;">
            <span class="ball" style="background: var(--gradient-2);">${r.number.toString().padStart(2, '0')}</span>
            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">${r.count}x</div>
        </div>`;
    });
    html += '</div></div>';
    
    // All numbers frequency grid
    html += '<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🔢</span> Bảng Tần Suất Toàn Bộ</div></div>';
    html += '<div class="number-grid">';
    
    const allFreq = a.all_frequency;
    const counts = Object.values(allFreq).map(f => f.count);
    const maxCount = Math.max(...counts);
    const minCount = Math.min(...counts);
    const threshold_hot = maxCount - (maxCount - minCount) * 0.2;
    const threshold_cold = minCount + (maxCount - minCount) * 0.2;
    
    for (let n = 1; n <= maxNum; n++) {
        const f = allFreq[n];
        if (!f) continue;
        let cls = '';
        if (f.count >= threshold_hot) cls = 'hot';
        else if (f.count <= threshold_cold) cls = 'cold';
        
        html += `<div class="number-cell ${cls}" title="Số ${n}: ${f.count} lần (${f.frequency}%)">
            ${n.toString().padStart(2, '0')}
            <span class="cell-count">${f.count}</span>
        </div>`;
    }
    html += '</div></div>';
    
    // Frequency Chart
    html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📊</span> Biểu Đồ Tần Suất</div></div>
        <canvas id="chart-freq-${type}" height="200"></canvas></div>`;
    
    container.innerHTML = html;
    
    // Render chart
    setTimeout(() => renderFreqChart(type, allFreq, maxNum), 100);
}

function renderFreqChart(type, allFreq, maxNum) {
    const canvasId = `chart-freq-${type}`;
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    // Destroy old chart
    if (charts[canvasId]) charts[canvasId].destroy();
    
    const labels = [];
    const values = [];
    const colors = [];
    
    const counts = Object.values(allFreq).map(f => f.count);
    const avg = counts.reduce((a, b) => a + b, 0) / counts.length;
    
    for (let n = 1; n <= maxNum; n++) {
        labels.push(n);
        const count = allFreq[n] ? allFreq[n].count : 0;
        values.push(count);
        
        if (count > avg * 1.15) {
            colors.push('rgba(236, 72, 153, 0.8)'); // hot pink
        } else if (count < avg * 0.85) {
            colors.push('rgba(6, 182, 212, 0.8)'); // cold cyan
        } else {
            colors.push('rgba(99, 102, 241, 0.6)'); // neutral
        }
    }
    
    charts[canvasId] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Số lần xuất hiện',
                data: values,
                backgroundColor: colors,
                borderRadius: 4,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12
                }
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.03)' }
                },
                y: {
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

// ---- Phase 5: Ultra Optimizer ----
function runPhase5(type) {
    const btn = document.getElementById('btn-phase5-' + type);
    const origText = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '⏳ Ultra Optimizer...'; btn.classList.add('pulsing');
    const container = document.getElementById('phase5-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card" style="border-color:#d4af37;"><div class="loading">
        <div class="spinner"></div><span class="loading-text">🏆 PHASE 5: Ultra Optimizer<br>
        Feature engineering + weight optimization + adaptive windows<br>Vui lòng chờ 3-5 phút...</span></div></div>`;
    fetch(`/api/phase5/${type}`, {method:'POST',headers:{'Content-Type':'application/json'}})
    .then(r => r.json()).then(data => {
        btn.disabled = false; btn.innerHTML = origText; btn.classList.remove('pulsing');
        if (data.error) { container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`; return; }
        renderPhase5(container, data, type); showToast('Phase 5 Ultra hoàn tất! 🏆', 'success');
    }).catch(err => { btn.disabled=false;btn.innerHTML=origText;btn.classList.remove('pulsing');
        container.innerHTML=`<div class="card"><div class="empty-state"><div class="icon">❌</div><p>${err.message}</p></div></div>`; });
}
function renderPhase5(container, data, type) {
    let html = '';
    const v = data.verdict || {};
    const score = v.score || 0;
    const best = v.best_strategy || {};
    html += `<div class="card" style="border-color:#d4af37;box-shadow:0 0 50px rgba(212,175,55,0.3);">
        <div class="card-header"><div class="card-title"><span class="icon">🏆</span> PHASE 5: ULTRA OPTIMIZER - KẾT QUẢ TỐI ƯU</div></div>
        <div style="text-align:center;margin:20px 0;">
            <div style="font-size:1.4rem;font-weight:700;color:#d4af37;">Best: ${best.name||'N/A'}</div>
            <div style="font-size:2.5rem;font-weight:900;color:#22c55e;margin:8px 0;">${best.avg||0}/6</div>
            <div style="font-size:1.2rem;font-weight:700;color:${best.improvement>0?'#22c55e':'#ef4444'};">${best.improvement>0?'+':''}${best.improvement||0}% so với random</div>
            <div style="font-size:0.85rem;color:var(--text-muted);margin-top:4px;">"${v.verdict||''}"</div>
        </div></div>`;
    if (data.next_prediction) {
        const np = data.next_prediction;
        html += `<div class="card" style="border-color:#22c55e;box-shadow:0 0 30px rgba(34,197,94,0.2);">
            <div class="card-header"><div class="card-title"><span class="icon">🔮</span> DỰ ĐOÁN TỐI ƯU KỲ TIẾP</div></div>
            <div style="text-align:center;margin:20px 0;">
                <div style="margin-bottom:12px;">${np.numbers ? np.numbers.map(n =>
                    `<span class="ball" style="font-size:1.4rem;width:58px;height:58px;line-height:58px;background:linear-gradient(135deg,#d4af37,#b8860b);box-shadow:0 4px 15px rgba(212,175,55,0.4);">${n.toString().padStart(2,'0')}</span>`).join('') : ''}</div>
                <div style="font-size:0.8rem;color:var(--text-muted);">${np.method||''}</div>
            </div></div>`;
    }
    if (v.strategy_ranking) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🏆</span> Xếp Hạng (Walk-Forward 200 kỳ)</div></div>
        <div class="data-table-wrap"><table class="data-table"><thead><tr>
            <th>#</th><th>Mô Hình</th><th>TB/6</th><th>Max</th><th>3+</th><th>4+</th><th>vs Random</th>
        </tr></thead><tbody>`;
        v.strategy_ranking.forEach((s, i) => {
            const g = s.improvement > 10;
            html += `<tr style="${i===0?'background:rgba(212,175,55,0.15);':g?'background:rgba(34,197,94,0.08);':''}">
                <td>${i===0?'🥇':i===1?'🥈':i===2?'🥉':(i+1)}</td>
                <td><strong>${s.name}</strong></td>
                <td><strong>${s.avg}</strong></td><td>${s.max}/6</td>
                <td>${s.match_3plus||0}</td><td>${s.match_4plus||0}</td>
                <td><span style="color:${s.improvement>0?'#22c55e':'#ef4444'};font-weight:700;">${s.improvement>0?'+':''}${s.improvement}%</span></td></tr>`;
        });
        html += '</tbody></table></div></div>';
    }
    if (data.adaptive_window && data.adaptive_window.window_scan) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📏</span> Adaptive Window Scan</div></div>
        <div class="data-table-wrap"><table class="data-table"><thead><tr><th>Window</th><th>TB/6</th><th>vs Random</th></tr></thead><tbody>`;
        data.adaptive_window.window_scan.forEach(w => {
            html += `<tr style="${w.window===data.adaptive_window.best_window?'background:rgba(212,175,55,0.15);':''}">
                <td>${w.window===data.adaptive_window.best_window?'⭐ ':' '}${w.window}</td>
                <td><strong>${w.avg}</strong></td>
                <td><span style="color:${w.imp>0?'#22c55e':'#ef4444'};">${w.imp>0?'+':''}${w.imp}%</span></td></tr>`;
        });
        html += '</tbody></table></div></div>';
    }
    if (v.evidence) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📝</span> Tổng Kết Phase 5</div></div>
        <div style="font-family:monospace;font-size:0.78rem;line-height:1.8;">`;
        v.evidence.forEach(e => { html += `<div style="color:${e.startsWith('+')?'#22c55e':'#ef4444'};">${e}</div>`; });
        html += '</div></div>';
    }
    container.innerHTML = html;
}

// ---- Phase 6: Deep Intelligence Engine ----
function runPhase6(type) {
    const btn = document.getElementById('btn-phase6-' + type);
    const origText = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '⏳ Deep Intelligence đang phân tích...'; btn.classList.add('pulsing');
    const container = document.getElementById('phase6-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card" style="border-color:#6366f1;"><div class="loading">
        <div class="spinner"></div><span class="loading-text">🧠 PHASE 6: Deep Intelligence Engine<br>
        HMM Regime · Bayesian Network · Simulated Annealing · Info Theory<br>
        Copula · Stacking · Run-Length · Freq Decomposition · Benford · Coverage<br>
        Walk-forward backtest 200 kỳ. Vui lòng chờ 3-5 phút.</span></div></div>`;
    fetch(`/api/phase6/${type}`, {method:'POST',headers:{'Content-Type':'application/json'}})
    .then(r => r.json()).then(data => {
        btn.disabled = false; btn.innerHTML = origText; btn.classList.remove('pulsing');
        if (data.error) { container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`; return; }
        renderPhase6(container, data, type); showToast('Phase 6 Deep Intelligence hoàn tất! 🧠', 'success');
    }).catch(err => { btn.disabled=false;btn.innerHTML=origText;btn.classList.remove('pulsing');
        container.innerHTML=`<div class="card"><div class="empty-state"><div class="icon">❌</div><p>${err.message}</p></div></div>`; });
}
function renderPhase6(container, data, type) {
    let html = '';
    const v = data.verdict || {};
    const score = v.score || 0;
    const best = v.best_strategy || {};
    const sColor = score >= 50 ? '#22c55e' : score >= 20 ? '#6366f1' : '#0ea5e9';
    // Verdict Card
    html += `<div class="card" style="border-color:${sColor};box-shadow:0 0 50px ${sColor}44;">
        <div class="card-header"><div class="card-title"><span class="icon">🧠</span> PHASE 6: DEEP INTELLIGENCE - KẾT QUẢ</div></div>
        <div style="text-align:center;margin:20px 0;">
            <div style="font-size:1.3rem;font-weight:700;color:${sColor};">Best: ${best.name||'N/A'}</div>
            <div style="font-size:2.8rem;font-weight:900;color:#22c55e;margin:8px 0;">${best.avg||0}/6</div>
            <div style="font-size:1.2rem;font-weight:700;color:${(best.improvement||0)>0?'#22c55e':'#ef4444'};">${(best.improvement||0)>0?'+':''}${best.improvement||0}% so với random</div>
            <div style="font-size:0.85rem;color:var(--text-muted);margin-top:6px;">"${v.verdict||''}"</div>
            <div style="font-size:0.8rem;color:var(--text-muted);">${v.pattern_count||0}/${v.total_tests||0} strategies beat random</div>
        </div></div>`;
    // Next Prediction
    if (data.next_prediction) {
        const np = data.next_prediction;
        html += `<div class="card" style="border-color:#22c55e;box-shadow:0 0 30px rgba(34,197,94,0.25);">
            <div class="card-header"><div class="card-title"><span class="icon">🔮</span> DỰ ĐOÁN KỲ TIẾP (Deep Intelligence)</div></div>
            <div style="text-align:center;margin:20px 0;">
                <div style="margin-bottom:14px;">${np.numbers ? np.numbers.map(n =>
                    `<span class="ball" style="font-size:1.5rem;width:62px;height:62px;line-height:62px;background:linear-gradient(135deg,#0ea5e9,#6366f1);box-shadow:0 4px 20px rgba(99,102,241,0.4);">${n.toString().padStart(2,'0')}</span>`).join('') : 'N/A'}</div>
                <div style="font-size:0.8rem;color:var(--text-muted);">${np.method||''}</div>
            </div>`;
        // Confidence bars
        if (np.score_distribution) {
            html += '<div style="margin-top:16px;"><strong>Confidence Scores (Top 15):</strong><div style="margin-top:10px;">';
            np.score_distribution.forEach(s => {
                const isSelected = np.numbers && np.numbers.includes(s.number);
                html += `<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                    <span class="ball small" style="${isSelected?'background:linear-gradient(135deg,#0ea5e9,#6366f1);box-shadow:0 2px 8px rgba(99,102,241,0.3);':''}">${s.number.toString().padStart(2,'0')}</span>
                    <div style="flex:1;height:8px;background:var(--bg-tertiary);border-radius:4px;overflow:hidden;">
                        <div style="height:100%;width:${s.confidence}%;background:linear-gradient(90deg,#0ea5e9,#6366f1);border-radius:4px;"></div></div>
                    <div style="font-size:0.72rem;min-width:50px;text-align:right;color:var(--text-muted);">${s.score} pts</div></div>`;
            });
            html += '</div></div>';
        }
        html += '</div>';
    }
    // Strategy Ranking Table
    if (v.strategy_ranking) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🏆</span> Xếp Hạng 10 Chiến Lược (Walk-Forward 200 kỳ)</div></div>
        <div class="data-table-wrap"><table class="data-table"><thead><tr>
            <th>#</th><th>Chiến Lược</th><th>TB/6</th><th>Max</th><th>3+</th><th>4+</th><th>vs Random</th>
        </tr></thead><tbody>`;
        v.strategy_ranking.forEach((s, i) => {
            const g = s.improvement > 10;
            html += `<tr style="${i===0?'background:rgba(99,102,241,0.15);':g?'background:rgba(34,197,94,0.08);':''}">
                <td>${i===0?'🥇':i===1?'🥈':i===2?'🥉':(i+1)}</td>
                <td><strong>${s.name}</strong></td>
                <td><strong>${s.avg}</strong></td><td>${s.max||'-'}/6</td>
                <td>${s.match_3plus||0}</td><td>${s.match_4plus||0}</td>
                <td><span style="color:${s.improvement>0?'#22c55e':'#ef4444'};font-weight:700;">${s.improvement>0?'+':''}${s.improvement}%</span></td></tr>`;
        });
        html += '</tbody></table></div></div>';
    }
    // Detail cards for interesting methods
    html += '<div class="grid-2">';
    // HMM Regime
    if (data.hmm_regime && data.hmm_regime.regime_breakdown) {
        const hm = data.hmm_regime;
        html += `<div class="prediction-card"><div class="model-name">📊 HMM Regime Detection</div>
            <div style="margin-top:8px;">
                <div>Hot: <strong style="color:#ef4444;">${hm.regime_breakdown.hot||0}</strong> numbers</div>
                <div>Neutral: <strong>${hm.regime_breakdown.neutral||0}</strong> numbers</div>
                <div>Cold: <strong style="color:#3b82f6;">${hm.regime_breakdown.cold||0}</strong> numbers</div>
                <div style="margin-top:6px;">TB: <strong>${hm.avg_matches}</strong>/6 (${hm.improvement>0?'+':''}${hm.improvement}%)</div>
            </div></div>`;
    }
    // Benford Evolution
    if (data.benford_evo && data.benford_evo.digit_analysis) {
        html += `<div class="prediction-card"><div class="model-name">📐 Benford Evolution</div><div style="margin-top:8px;">`;
        for (const [digit, info] of Object.entries(data.benford_evo.digit_analysis)) {
            const dev = info.deviation || 0;
            html += `<div style="display:flex;justify-content:space-between;font-size:0.78rem;">
                <span>Digit ${digit}</span>
                <span>Actual: ${info.actual_pct}%</span>
                <span>Benford: ${info.benford_pct}%</span>
                <span style="color:${Math.abs(dev)>5?'#ef4444':'#22c55e'};font-weight:700;">${dev>0?'+':''}${dev}%</span></div>`;
        }
        html += `<div style="margin-top:6px;">TB: <strong>${data.benford_evo.avg_matches}</strong>/6 (${data.benford_evo.improvement>0?'+':''}${data.benford_evo.improvement}%)</div></div></div>`;
    }
    // Stacking
    if (data.stacking && data.stacking.predictor_quality) {
        html += `<div class="prediction-card"><div class="model-name">📚 Stacking Meta-Learner</div>
            <div style="margin-top:8px;">
                <div>Base predictors: <strong>${(data.stacking.base_predictors||[]).length}</strong></div>`;
        for (const [name, q] of Object.entries(data.stacking.predictor_quality)) {
            html += `<div style="font-size:0.78rem;">${name}: ${q} matches last draw</div>`;
        }
        html += `<div style="margin-top:6px;">TB: <strong>${data.stacking.avg_matches}</strong>/6 (${data.stacking.improvement>0?'+':''}${data.stacking.improvement}%)</div></div></div>`;
    }
    // Simulated Annealing
    if (data.simulated_annealing) {
        const sa = data.simulated_annealing;
        html += `<div class="prediction-card"><div class="model-name">🔥 Simulated Annealing</div>
            <div style="margin-top:8px;">
                <div>TB: <strong>${sa.avg_matches}</strong>/6</div>
                <div>Max: <strong>${sa.max_matches}</strong>/6</div>
                <div>vs Random: <span style="color:${sa.improvement>0?'#22c55e':'#ef4444'};font-weight:700;">${sa.improvement>0?'+':''}${sa.improvement}%</span></div>
            </div></div>`;
    }
    // Info Theory
    if (data.info_theory) {
        html += `<div class="prediction-card"><div class="model-name">🧮 Information Theory</div>
            <div style="margin-top:8px;">
                <div>Conditional Entropy selector</div>
                <div>TB: <strong>${data.info_theory.avg_matches}</strong>/6</div>
                <div>vs Random: <span style="color:${data.info_theory.improvement>0?'#22c55e':'#ef4444'};font-weight:700;">${data.info_theory.improvement>0?'+':''}${data.info_theory.improvement}%</span></div>
            </div></div>`;
    }
    // Run-Length
    if (data.run_length) {
        html += `<div class="prediction-card"><div class="model-name">📏 Run-Length Predictor</div>
            <div style="margin-top:8px;">
                <div>Turning point detection</div>
                <div>TB: <strong>${data.run_length.avg_matches}</strong>/6</div>
                <div>vs Random: <span style="color:${data.run_length.improvement>0?'#22c55e':'#ef4444'};font-weight:700;">${data.run_length.improvement>0?'+':''}${data.run_length.improvement}%</span></div>
            </div></div>`;
    }
    html += '</div>';
    // Summary table of all 10
    const methods = ['hmm_regime','bayesian_net','simulated_annealing','info_theory','copula','stacking','run_length','freq_decomp','benford_evo','coverage_opt'];
    const icons = {'hmm_regime':'📊','bayesian_net':'🎯','simulated_annealing':'🔥','info_theory':'🧮','copula':'🔗','stacking':'📚','run_length':'📏','freq_decomp':'📈','benford_evo':'📐','coverage_opt':'🎪'};
    html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📋</span> 10 Phương Pháp Chi Tiết</div></div>
    <div class="data-table-wrap"><table class="data-table"><thead><tr>
        <th></th><th>Phương Pháp</th><th>TB/6</th><th>Max</th><th>vs Random</th><th>Tests</th>
    </tr></thead><tbody>`;
    methods.forEach(key => {
        const t = data[key]; if (!t) return;
        const imp = t.improvement || 0;
        html += `<tr style="${imp>15?'background:rgba(34,197,94,0.1);':''}">
            <td>${icons[key]||'📊'}</td><td><strong>${t.name||key}</strong></td>
            <td><strong>${t.avg_matches||'-'}</strong></td><td>${t.max_matches||'-'}/6</td>
            <td><span style="color:${imp>0?'#22c55e':'#ef4444'};font-weight:700;">${imp>0?'+':''}${imp}%</span></td>
            <td>${t.tests||'-'}</td></tr>`;
    });
    html += '</tbody></table></div></div>';
    // Evidence
    if (v.evidence) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📝</span> Tổng Kết Phase 6</div></div>
        <div style="font-family:monospace;font-size:0.78rem;line-height:1.8;">`;
        v.evidence.forEach(e => { html += `<div style="color:${e.startsWith('+')?'#22c55e':'#ef4444'};">${e}</div>`; });
        html += '</div></div>';
    }
    container.innerHTML = html;
}

// ---- Phase 7: Ultimate Predictor ----
function runPhase7(type) {
    const btn = document.getElementById('btn-phase7-' + type);
    const origText = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '⏳ Ultimate Predictor...'; btn.classList.add('pulsing');
    const container = document.getElementById('phase7-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card" style="border-color:#f43f5e;box-shadow:0 0 40px rgba(244,63,94,0.3);"><div class="loading">
        <div class="spinner"></div><span class="loading-text">👑 PHASE 7: Ultimate Predictor<br>
        Cross-Phase Meta · Genetic Evolver · Cluster Match · Recurrence<br>
        Entropy Min · Multi-Scale · Graph Walk · Diff Evolution · Temporal Gradient<br>
        ULTIMATE FUSION (70+ signals). Vui lòng chờ 5-8 phút.</span></div></div>`;
    fetch(`/api/phase7/${type}`, {method:'POST',headers:{'Content-Type':'application/json'}})
    .then(r => r.json()).then(data => {
        btn.disabled = false; btn.innerHTML = origText; btn.classList.remove('pulsing');
        if (data.error) { container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`; return; }
        renderPhase7(container, data, type); showToast('Phase 7 ULTIMATE hoàn tất! 👑', 'success');
    }).catch(err => { btn.disabled=false;btn.innerHTML=origText;btn.classList.remove('pulsing');
        container.innerHTML=`<div class="card"><div class="empty-state"><div class="icon">❌</div><p>${err.message}</p></div></div>`; });
}
function renderPhase7(container, data, type) {
    let html = '';
    const v = data.verdict || {};
    const best = v.best_strategy || {};
    const sColor = '#f43f5e';
    // VERDICT
    html += `<div class="card" style="border-color:${sColor};box-shadow:0 0 60px rgba(244,63,94,0.3);background:linear-gradient(135deg, rgba(244,63,94,0.05), rgba(124,58,237,0.05));">
        <div class="card-header"><div class="card-title"><span class="icon">👑</span> PHASE 7: ULTIMATE PREDICTOR - KẾT QUẢ CUỐI CÙNG</div></div>
        <div style="text-align:center;margin:20px 0;">
            <div style="font-size:1.3rem;font-weight:700;color:${sColor};">Best Strategy: ${best.name||'N/A'}</div>
            <div style="font-size:3rem;font-weight:900;background:linear-gradient(135deg,#f43f5e,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:8px 0;">${best.avg||0}/6</div>
            <div style="font-size:1.2rem;font-weight:700;color:${(best.improvement||0)>0?'#22c55e':'#ef4444'};">${(best.improvement||0)>0?'+':''}${best.improvement||0}% so với random</div>
            <div style="font-size:0.85rem;color:var(--text-muted);margin-top:6px;">"${v.verdict||''}"</div>
            <div style="font-size:0.8rem;color:var(--text-muted);">${v.pattern_count||0}/${v.total_tests||0} strategies beat random</div>
        </div></div>`;
    // 3 PREDICTION SETS
    if (data.next_prediction) {
        const np = data.next_prediction;
        html += `<div class="card" style="border-color:#22c55e;box-shadow:0 0 40px rgba(34,197,94,0.3);">
            <div class="card-header"><div class="card-title"><span class="icon">🔮</span> DỰ ĐOÁN KỲ TIẾP (70+ signals kết hợp)</div></div>
            <div class="prediction-grid">
                <div class="prediction-card" style="border-color:#f43f5e;box-shadow:0 0 20px rgba(244,63,94,0.2);">
                    <div class="model-name">👑 Primary (TOP 6)</div>
                    <div class="numbers">${np.primary ? np.primary.map(n => `<span class="ball" style="background:linear-gradient(135deg,#f43f5e,#7c3aed);font-size:1.3rem;width:56px;height:56px;line-height:56px;box-shadow:0 4px 15px rgba(244,63,94,0.4);">${n.toString().padStart(2,'0')}</span>`).join('') : ''}</div>
                </div>
                <div class="prediction-card" style="border-color:#6366f1;">
                    <div class="model-name">🥈 Secondary (Alt 1)</div>
                    <div class="numbers">${np.secondary ? np.secondary.map(n => `<span class="ball" style="background:linear-gradient(135deg,#6366f1,#0ea5e9);">${n.toString().padStart(2,'0')}</span>`).join('') : ''}</div>
                </div>
                <div class="prediction-card">
                    <div class="model-name">🥉 Tertiary (Alt 2)</div>
                    <div class="numbers">${np.tertiary ? np.tertiary.map(n => `<span class="ball">${n.toString().padStart(2,'0')}</span>`).join('') : ''}</div>
                </div>
            </div>`;
        if (np.score_distribution) {
            html += '<div style="margin-top:20px;"><strong>Confidence Scores (Top 18):</strong><div style="margin-top:10px;">';
            np.score_distribution.forEach(s => {
                const isPrimary = np.primary && np.primary.includes(s.number);
                const isSec = np.secondary && np.secondary.includes(s.number);
                const bg = isPrimary ? 'background:linear-gradient(135deg,#f43f5e,#7c3aed);box-shadow:0 2px 10px rgba(244,63,94,0.3);' :
                           isSec ? 'background:linear-gradient(135deg,#6366f1,#0ea5e9);' : '';
                html += `<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                    <span class="ball small" style="${bg}">${s.number.toString().padStart(2,'0')}</span>
                    <div style="flex:1;height:8px;background:var(--bg-tertiary);border-radius:4px;overflow:hidden;">
                        <div style="height:100%;width:${s.confidence}%;background:linear-gradient(90deg,#f43f5e,#7c3aed);border-radius:4px;"></div></div>
                    <div style="font-size:0.72rem;min-width:50px;text-align:right;color:var(--text-muted);">${s.score} pts</div></div>`;
            });
            html += '</div></div>';
        }
        html += `<div style="margin-top:10px;font-size:0.75rem;color:var(--text-muted);text-align:center;">${np.method||''}</div></div>`;
    }
    // STRATEGY RANKING
    if (v.strategy_ranking) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🏆</span> Xếp Hạng 10 Chiến Lược Phase 7 (Walk-Forward 200 kỳ)</div></div>
        <div class="data-table-wrap"><table class="data-table"><thead><tr>
            <th>#</th><th>Chiến Lược</th><th>TB/6</th><th>Max</th><th>3+</th><th>4+</th><th>vs Random</th>
        </tr></thead><tbody>`;
        v.strategy_ranking.forEach((s, i) => {
            const g = s.improvement > 10;
            html += `<tr style="${i===0?'background:rgba(244,63,94,0.12);':g?'background:rgba(34,197,94,0.08);':''}">
                <td>${i===0?'👑':i===1?'🥈':i===2?'🥉':(i+1)}</td>
                <td><strong>${s.name}</strong></td>
                <td><strong>${s.avg}</strong></td><td>${s.max||'-'}/6</td>
                <td>${s.match_3plus||0}</td><td>${s.match_4plus||0}</td>
                <td><span style="color:${s.improvement>0?'#22c55e':'#ef4444'};font-weight:700;">${s.improvement>0?'+':''}${s.improvement}%</span></td></tr>`;
        });
        html += '</tbody></table></div></div>';
    }
    // Detail cards
    html += '<div class="grid-2">';
    const methods = ['meta_ensemble','genetic_evolver','cluster_match','recurrence','entropy_min','multi_scale','graph_walk','diff_evolution','temporal_grad','ultimate_fusion'];
    const icons = {'meta_ensemble':'🔀','genetic_evolver':'🧬','cluster_match':'🎯','recurrence':'🔄','entropy_min':'📉','multi_scale':'📊','graph_walk':'🕸️','diff_evolution':'🧪','temporal_grad':'📈','ultimate_fusion':'👑'};
    methods.forEach(key => {
        const t = data[key]; if (!t) return;
        const imp = t.improvement || 0;
        html += `<div class="prediction-card" style="${imp>15?'border-color:#22c55e;':''}">
            <div class="model-name">${icons[key]||'📊'} ${t.name||key}</div>
            <div style="margin-top:6px;">
                <div>TB: <strong>${t.avg_matches||'-'}</strong>/6 · Max: <strong>${t.max_matches||'-'}</strong>/6</div>
                <div>3+ matches: <strong>${t.match_3plus||0}</strong> · 4+: <strong>${t.match_4plus||0}</strong></div>
                <div>vs Random: <span style="color:${imp>0?'#22c55e':'#ef4444'};font-weight:700;">${imp>0?'+':''}${imp}%</span></div>`;
        if (t.evolved_weights) { html += `<div style="font-size:0.7rem;color:var(--text-muted);margin-top:4px;">Evolved weights: [${t.evolved_weights.join(', ')}]</div>`; }
        html += '</div></div>';
    });
    html += '</div>';
    // Evidence
    if (v.evidence) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📝</span> Tổng Kết Phase 7</div></div>
        <div style="font-family:monospace;font-size:0.78rem;line-height:1.8;">`;
        v.evidence.forEach(e => { html += `<div style="color:${e.startsWith('+')?'#22c55e':'#ef4444'};">${e}</div>`; });
        html += '</div></div>';
    }
    container.innerHTML = html;
}

// ---- Phase 4: Exploit Engine ----
function runPhase4(type) {
    const btn = document.getElementById('btn-phase4-' + type);
    const origText = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '⏳ Exploit Engine đang chạy...'; btn.classList.add('pulsing');
    const container = document.getElementById('phase4-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card" style="border-color:#ea580c;"><div class="loading">
        <div class="spinner"></div><span class="loading-text">🎯 PHASE 4: Exploit Engine<br>
        10 chiến lược kết hợp + walk-forward backtest 200 kỳ<br>Vui lòng chờ 3-5 phút...</span></div></div>`;
    fetch(`/api/phase4/${type}`, {method:'POST',headers:{'Content-Type':'application/json'}})
    .then(r => r.json()).then(data => {
        btn.disabled = false; btn.innerHTML = origText; btn.classList.remove('pulsing');
        if (data.error) { container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`; return; }
        renderPhase4(container, data, type); showToast('Phase 4 Exploit hoàn tất! 🎯', 'success');
    }).catch(err => { btn.disabled=false;btn.innerHTML=origText;btn.classList.remove('pulsing');
        container.innerHTML=`<div class="card"><div class="empty-state"><div class="icon">❌</div><p>${err.message}</p></div></div>`; });
}
function renderPhase4(container, data, type) {
    let html = '';
    const v = data.verdict || {};
    const score = v.score || 0;
    const sColor = score >= 50 ? '#22c55e' : score >= 20 ? '#eab308' : '#ea580c';
    // Verdict
    html += `<div class="card" style="border-color:${sColor};box-shadow:0 0 40px ${sColor}33;">
        <div class="card-header"><div class="card-title"><span class="icon">🎯</span> PHASE 4: EXPLOIT ENGINE - KẾT QUẢ</div></div>
        <div style="text-align:center;margin:20px 0;">
            <div style="font-size:3rem;font-weight:900;color:${sColor};">${score}%</div>
            <div style="font-size:1.1rem;font-weight:700;color:${sColor};margin-top:8px;">"${v.verdict||''}"</div>
            <div style="font-size:0.85rem;color:var(--text-muted);">${v.pattern_count||0}/${v.total_tests||0} strategies beat random</div>
        </div></div>`;
    // Next Prediction
    if (data.next_prediction) {
        const np = data.next_prediction;
        html += `<div class="card" style="border-color:#22c55e;box-shadow:0 0 30px rgba(34,197,94,0.2);">
            <div class="card-header"><div class="card-title"><span class="icon">🔮</span> DỰ ĐOÁN KỲ TIẾP THEO (Exploit Engine)</div></div>
            <div style="text-align:center;margin:20px 0;">
                <div style="margin-bottom:12px;">${np.numbers ? np.numbers.map(n => 
                    `<span class="ball" style="font-size:1.3rem;width:55px;height:55px;line-height:55px;">${n.toString().padStart(2,'0')}</span>`).join('') : 'N/A'}</div>
                <div style="font-size:0.8rem;color:var(--text-muted);">${np.method||''}</div>
            </div></div>`;
    }
    // Strategy Ranking
    if (v.strategy_ranking) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🏆</span> Xếp Hạng Chiến Lược (Walk-Forward Backtest)</div></div>
        <div class="data-table-wrap"><table class="data-table"><thead><tr>
            <th>#</th><th>Chiến Lược</th><th>TB Đúng/6</th><th>So Với Random</th>
        </tr></thead><tbody>`;
        v.strategy_ranking.forEach((s, i) => {
            const imp = s.improvement || 0;
            const isGood = imp > 10;
            html += `<tr style="${isGood?'background:rgba(34,197,94,0.1);':''}">
                <td><strong>${i+1}</strong></td><td>${s.name}</td>
                <td><strong>${s.avg}</strong></td>
                <td><span style="color:${imp>0?'#22c55e':'#ef4444'};font-weight:700;">${imp>0?'+':''}${imp}%</span></td></tr>`;
        });
        html += '</tbody></table></div></div>';
    }
    // Summary table
    const tests = ['lcg_exploit','fft_exploit','markov3','conditional','hotzone','delta','position','pair_net','super_ensemble'];
    html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📋</span> 9 Chiến Lược Chi Tiết</div></div>
    <div class="data-table-wrap"><table class="data-table"><thead><tr>
        <th>Chiến Lược</th><th>TB Đúng/6</th><th>Max</th><th>So Với Random</th><th>Tests</th>
    </tr></thead><tbody>`;
    tests.forEach(key => {
        const t = data[key]; if (!t) return;
        const imp = t.improvement || 0;
        html += `<tr style="${imp>15?'background:rgba(34,197,94,0.1);':''}">
            <td><strong>${t.name||key}</strong></td>
            <td><strong>${t.avg_matches||'-'}</strong></td>
            <td>${t.max_matches||'-'}/6</td>
            <td><span style="color:${imp>0?'#22c55e':'#ef4444'};font-weight:700;">${imp>0?'+':''}${imp}%</span></td>
            <td>${t.tests||'-'}</td></tr>`;
    });
    html += '</tbody></table></div></div>';
    // LCG detail
    if (data.lcg_exploit) {
        const lc = data.lcg_exploit;
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🔑</span> LCG Seed Exploit Details</div></div>
            <div style="padding:12px;">
                <div>Best match: <strong style="color:${lc.best_match>=6?'#ef4444':'#22c55e'};">${lc.best_match}/${lc.out_of}</strong></div>
                <div style="font-size:0.8rem;word-break:break-all;margin-top:6px;">${JSON.stringify(lc.best_config)}</div>
                <div>Seeds tested: ${lc.seeds_tested}</div>
                ${lc.predicted_next && lc.predicted_next.length ? `<div style="margin-top:12px;"><strong>LCG predicted next:</strong> ${lc.predicted_next.map(n => 
                    `<span class="ball small">${n.toString().padStart(2,'0')}</span>`).join('')}</div>` : ''}
            </div></div>`;
    }
    // Evidence
    if (v.evidence) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📝</span> Tổng Kết</div></div>
        <div style="font-family:monospace;font-size:0.78rem;line-height:1.8;">`;
        v.evidence.forEach(e => { html += `<div style="color:${e.startsWith('+')?'#22c55e':'#ef4444'};">${e}</div>`; });
        html += '</div></div>';
    }
    container.innerHTML = html;
}

// ---- Phase 3: Forensic ----
function runPhase3(type) {
    const btn = document.getElementById('btn-phase3-' + type);
    const origText = btn.innerHTML;
    btn.disabled = true; btn.innerHTML = '⏳ Pháp y số học đang chạy...'; btn.classList.add('pulsing');
    const container = document.getElementById('phase3-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card" style="border-color:#059669;"><div class="loading">
        <div class="spinner"></div><span class="loading-text">🔍 PHASE 3: Benford's Law, Birthday Spacing, Seed Hunter...<br>
        Brute-force 16,000 seeds. Vui lòng chờ 2-3 phút.</span></div></div>`;
    fetch(`/api/phase3/${type}`, {method:'POST',headers:{'Content-Type':'application/json'}})
    .then(r => r.json()).then(data => {
        btn.disabled = false; btn.innerHTML = origText; btn.classList.remove('pulsing');
        if (data.error) { container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`; return; }
        renderPhase3(container, data, type); showToast('Phase 3 Forensic hoàn tất! 🔍', 'success');
    }).catch(err => { btn.disabled=false;btn.innerHTML=origText;btn.classList.remove('pulsing');
        container.innerHTML=`<div class="card"><div class="empty-state"><div class="icon">❌</div><p>${err.message}</p></div></div>`; });
}
function renderPhase3(container, data, type) {
    let html = '';
    const v = data.verdict || {};
    const score = v.score || 0;
    const sColor = score >= 60 ? '#ef4444' : score >= 30 ? '#eab308' : '#059669';
    html += `<div class="card" style="border-color:${sColor};box-shadow:0 0 40px ${sColor}33;">
        <div class="card-header"><div class="card-title"><span class="icon">🔍</span> PHASE 3: PHÁP Y SỐ HỌC - KẾT QUẢ</div></div>
        <div style="text-align:center;margin:20px 0;">
            <div style="font-size:3rem;font-weight:900;color:${sColor};">${score}%</div>
            <div style="font-size:1.1rem;font-weight:700;color:${sColor};margin-top:8px;">"${v.verdict||''}"</div>
            <div style="font-size:0.85rem;color:var(--text-muted);">${v.pattern_count||0}/${v.total_tests||0} phát hiện bất thường</div>
        </div></div>`;
    // Summary table
    const tests = ['benford','birthday','streaks','digits','position_bias','consecutive','pairs','sum_theory','coupon','seed_hunt'];
    const icons = {'benford':'📊','birthday':'🎂','streaks':'🔥','digits':'🔢','position_bias':'📍',
        'consecutive':'🔗','pairs':'👥','sum_theory':'📐','coupon':'🎟️','seed_hunt':'🔑'};
    html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📋</span> 10 Phương Pháp Pháp Y</div></div>
    <div class="data-table-wrap"><table class="data-table"><thead><tr><th></th><th>Phương Pháp</th><th>Kết Quả</th><th>Kết Luận</th></tr></thead><tbody>`;
    tests.forEach(key => {
        const t = data[key]; if (!t) return;
        const isPat = t.is_pattern;
        html += `<tr style="${isPat?'background:rgba(239,68,68,0.1);':''}">
            <td>${icons[key]||'📊'}</td><td><strong>${t.name||key}</strong></td>
            <td><span style="padding:3px 10px;border-radius:12px;font-weight:700;font-size:0.8rem;
            background:${isPat?'rgba(239,68,68,0.2)':'rgba(34,197,94,0.2)'};
            color:${isPat?'#ef4444':'#22c55e'};">${isPat?'🔴 BẤT THƯỜNG':'🟢 BÌNH THƯỜNG'}</span></td>
            <td style="font-size:0.8rem;color:var(--text-muted);">${t.conclusion||''}</td></tr>`;
    });
    html += '</tbody></table></div></div>';
    html += '<div class="grid-2">';
    // Streaks
    if (data.streaks) {
        const s = data.streaks;
        html += `<div class="prediction-card"><div class="model-name">🔥 Streak Analysis</div>
            <div style="margin-top:8px;">
                <div>Hot streak max: <strong>${s.max_hot_streak.length}</strong> kỳ liên tiếp (số <span class="ball small">${s.max_hot_streak.number.toString().padStart(2,'0')}</span>) - Expected: ~${s.max_hot_streak.expected}</div>
                <div>Cold streak max: <strong>${s.max_cold_streak.length}</strong> kỳ vắng mặt (số <span class="ball small">${s.max_cold_streak.number.toString().padStart(2,'0')}</span>) - Expected: ~${s.max_cold_streak.expected}</div>
                <div>${s.num_anomalous} số bất thường</div>
            </div></div>`;
    }
    // Pairs
    if (data.pairs) {
        const p = data.pairs;
        html += `<div class="prediction-card"><div class="model-name">👥 Pair Hotspot</div>
            <div style="margin-top:8px;">
                <div>Max sigma: <strong style="color:${p.max_sigma>4?'#ef4444':'#22c55e'};">${p.max_sigma}σ</strong></div>
                <div>Cặp nóng nhất:</div>
                ${p.hottest_pairs ? p.hottest_pairs.slice(0,3).map(h => 
                    `<div><span class="ball small">${h.pair[0].toString().padStart(2,'0')}</span><span class="ball small">${h.pair[1].toString().padStart(2,'0')}</span> = ${h.count}x (${h.sigma}σ)</div>`
                ).join('') : ''}
            </div></div>`;
    }
    // Consecutive
    if (data.consecutive) {
        const c = data.consecutive;
        html += `<div class="prediction-card"><div class="model-name">🔗 Consecutive</div>
            <div style="margin-top:8px;">
                <div>TB cặp liên tiếp: <strong>${c.avg_consecutive_pairs}</strong> (expected: ${c.expected_consecutive})</div>
                <div>Chênh lệch: <strong style="color:${c.deviation_pct>15?'#ef4444':'#22c55e'};">${c.deviation_pct}%</strong></div>
                <div>Kỳ có 3+ liên tiếp: ${c.draws_with_3plus}</div>
            </div></div>`;
    }
    // Seed Hunter
    if (data.seed_hunt) {
        const sh = data.seed_hunt;
        html += `<div class="prediction-card"><div class="model-name">🔑 Seed Hunter</div>
            <p style="font-size:0.75rem;color:var(--text-muted);">Brute-force ${sh.seeds_tested}</p>
            <div style="margin-top:8px;">
                <div>Best match: <strong style="color:${sh.best_match>=5?'#ef4444':'#22c55e'};">${sh.best_match}/${sh.out_of}</strong></div>
                <div style="font-size:0.7rem;word-break:break-all;">${sh.best_algorithm}</div>
                <div>Random expected: ${sh.expected_random_matches}</div>
            </div></div>`;
    }
    // Sum vs Theory
    if (data.sum_theory) {
        const st = data.sum_theory;
        html += `<div class="prediction-card"><div class="model-name">📐 Sum vs Theory</div>
            <div style="margin-top:8px;">
                <div>Mean: <strong>${st.actual_mean}</strong> (exp: ${st.expected_mean})</div>
                <div>Deviation: <strong>${st.mean_deviation_sigma}σ</strong></div>
                <div>Skewness: ${st.skewness} | Kurtosis: ${st.kurtosis}</div>
            </div></div>`;
    }
    // Coupon
    if (data.coupon) {
        const cp = data.coupon;
        html += `<div class="prediction-card"><div class="model-name">🎟️ Coupon Collector</div>
            <div style="margin-top:8px;">
                <div>Phủ hết tại kỳ: <strong>#${cp.draws_to_complete}</strong> (expected: ~${cp.expected_draws})</div>
                <div>Chênh lệch: ${cp.deviation_pct}%</div>
                ${cp.milestones ? `<div style="font-size:0.75rem;">50%: kỳ #${cp.milestones['50pct']} | 90%: kỳ #${cp.milestones['90pct']}</div>` : ''}
            </div></div>`;
    }
    html += '</div>';
    // Evidence
    if (v.evidence) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📝</span> Tổng Kết Pháp Y</div></div>
        <div style="font-family:monospace;font-size:0.8rem;line-height:1.8;">`;
        v.evidence.forEach(e => { html += `<div style="color:${e.startsWith('+')?'#ef4444':'#22c55e'};">${e}</div>`; });
        html += '</div></div>';
    }
    container.innerHTML = html;
}

// ---- Phase 2: Advanced Cracking ----
function runPhase2(type) {
    const btn = document.getElementById('btn-phase2-' + type);
    const origText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '⏳ Phase 2 advanced analysis...';
    btn.classList.add('pulsing');
    
    const container = document.getElementById('phase2-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card" style="border-color: #7c3aed;"><div class="loading">
        <div class="spinner"></div>
        <span class="loading-text">🚀 PHASE 2: Chaos Theory, Genetic Algorithm, Deep GRU,<br>
        Compression, Cross-Lottery, Wavelets, Symbolic Regression...<br>
        10 phương pháp nâng cao. Vui lòng chờ 1-2 phút.</span>
    </div></div>`;
    
    fetch(`/api/phase2/${type}`, {method:'POST',headers:{'Content-Type':'application/json'}})
    .then(r => r.json())
    .then(data => {
        btn.disabled = false; btn.innerHTML = origText; btn.classList.remove('pulsing');
        if (data.error) { container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`; return; }
        renderPhase2(container, data, type);
        showToast('Phase 2 hoàn tất! 🚀', 'success');
    })
    .catch(err => {
        btn.disabled = false; btn.innerHTML = origText; btn.classList.remove('pulsing');
        container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">❌</div><p>${err.message}</p></div></div>`;
    });
}

function renderPhase2(container, data, type) {
    let html = '';
    const v = data.verdict || {};
    const score = v.score || 0;
    const sColor = score >= 60 ? '#22c55e' : score >= 30 ? '#eab308' : '#7c3aed';
    
    html += `<div class="card" style="border-color:${sColor};box-shadow:0 0 40px ${sColor}33;">
        <div class="card-header"><div class="card-title"><span class="icon">🚀</span> PHASE 2: KHOA HỌC NÂNG CAO - KẾT QUẢ</div></div>
        <div style="text-align:center;margin:20px 0;">
            <div style="font-size:3rem;font-weight:900;color:${sColor};">${score}%</div>
            <div style="font-size:1.1rem;font-weight:700;color:${sColor};margin-top:8px;">"${v.verdict||''}"</div>
            <div style="font-size:0.85rem;color:var(--text-muted);">${v.pattern_count||0}/${v.total_tests||0} phát hiện pattern</div>
        </div>
    </div>`;
    
    // Summary table
    const tests = ['chaos','state_space','genetic','deep_gru','compression',
                   'cross_lottery','mutual_info','wavelet','symbolic','meta_predictor'];
    const icons = {'chaos':'🌀','state_space':'🔄','genetic':'🧬','deep_gru':'🧠','compression':'📦',
                   'cross_lottery':'🔀','mutual_info':'ℹ️','wavelet':'〰️','symbolic':'📐','meta_predictor':'🎯'};
    
    html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📋</span> 10 Phương Pháp Nâng Cao</div></div>
    <div class="data-table-wrap"><table class="data-table"><thead><tr>
        <th></th><th>Phương Pháp</th><th>Kết Quả</th><th>Kết Luận</th>
    </tr></thead><tbody>`;
    tests.forEach(key => {
        const t = data[key];
        if (!t) return;
        const isPat = t.is_pattern;
        html += `<tr style="${isPat?'background:rgba(239,68,68,0.1);':''}">
            <td>${icons[key]||'📊'}</td><td><strong>${t.name||key}</strong></td>
            <td><span style="padding:3px 10px;border-radius:12px;font-weight:700;font-size:0.8rem;
            background:${isPat?'rgba(239,68,68,0.2)':'rgba(34,197,94,0.2)'};
            color:${isPat?'#ef4444':'#22c55e'};">${isPat?'🔴 PATTERN':'🟢 RANDOM'}</span></td>
            <td style="font-size:0.8rem;color:var(--text-muted);">${t.conclusion||''}</td></tr>`;
    });
    html += '</tbody></table></div></div>';
    
    // Detail cards
    html += '<div class="grid-2">';
    
    // Chaos
    if (data.chaos) {
        const c = data.chaos;
        html += `<div class="prediction-card"><div class="model-name">🌀 Chaos Theory</div>
            <p style="font-size:0.8rem;color:var(--text-muted);">Lyapunov Exponent - chaos vs random</p>
            <div style="margin-top:8px;">
                <div>Lyapunov: <strong>${c.lyapunov_exponent}</strong></div>
                <div>Shuffled: <strong>${c.shuffled_lyapunov}</strong></div>
                <div>Diff: <strong style="color:${c.difference>0.3?'#ef4444':'#22c55e'};">${c.difference}</strong></div>
                <div style="font-size:0.75rem;margin-top:4px;color:var(--accent-4);">${c.interpretation}</div>
            </div></div>`;
    }
    
    // Compression
    if (data.compression) {
        const cp = data.compression;
        html += `<div class="prediction-card"><div class="model-name">📦 Kolmogorov Complexity</div>
            <p style="font-size:0.8rem;color:var(--text-muted);">If compressible → hidden pattern</p>
            <div style="margin-top:8px;">
                <div>Data ratio: <strong>${cp.compression_ratio}</strong></div>
                <div>Random ratio: <strong>${cp.random_ratio}</strong></div>
                <div>Pattern ratio: <strong>${cp.pattern_ratio}</strong></div>
                <div>Diff vs random: <strong style="color:${cp.difference>0.05?'#ef4444':'#22c55e'};">${cp.difference}</strong></div>
            </div></div>`;
    }
    
    // Genetic
    if (data.genetic) {
        const g = data.genetic;
        html += `<div class="prediction-card"><div class="model-name">🧬 Genetic Algorithm</div>
            <p style="font-size:0.8rem;color:var(--text-muted);">Evolved prediction formulas</p>
            <div style="margin-top:8px;font-family:monospace;font-size:0.7rem;word-break:break-all;">${g.best_formula||'N/A'}</div>
            <div style="margin-top:6px;">Improvement: <strong style="color:${g.improvement>10?'#22c55e':'#ef4444'};">${g.improvement}%</strong></div>
        </div>`;
    }
    
    // Deep GRU
    if (data.deep_gru) {
        const d = data.deep_gru;
        html += `<div class="prediction-card"><div class="model-name">🧠 Deep GRU (3-layer)</div>
            <p style="font-size:0.8rem;color:var(--text-muted);">96 neurons, Leaky ReLU</p>
            <div style="margin-top:8px;">
                <div>Avg matches: <strong>${d.avg_matches}</strong>/6</div>
                <div>Random: ${d.random_expected}/6</div>
                <div>Improvement: <strong style="color:${d.improvement>20?'#22c55e':'#ef4444'};">${d.improvement>0?'+':''}${d.improvement}%</strong></div>
            </div></div>`;
    }
    
    // Cross-Lottery
    if (data.cross_lottery) {
        const cl = data.cross_lottery;
        html += `<div class="prediction-card"><div class="model-name">🔀 Cross-Lottery</div>
            <p style="font-size:0.8rem;color:var(--text-muted);">Mega↔Power correlation</p>
            <div style="margin-top:8px;">
                <div>Sum correlation: <strong>${cl.sum_correlation}</strong></div>
                <div>Number overlap: <strong>${cl.avg_number_overlap}</strong> (expected: ${cl.expected_overlap})</div>
            </div></div>`;
    }
    
    // Meta Predictor
    if (data.meta_predictor) {
        const mp = data.meta_predictor;
        html += `<div class="prediction-card"><div class="model-name">🎯 Meta-Predictor</div>
            <p style="font-size:0.8rem;color:var(--text-muted);">Combined 5 signals</p>
            <div style="margin-top:8px;">
                <div>Avg matches: <strong>${mp.avg_matches}</strong>/6</div>
                <div>Random: ${mp.random_expected}/6</div>
                <div>Improvement: <strong style="color:${mp.improvement>20?'#22c55e':'#ef4444'};">${mp.improvement>0?'+':''}${mp.improvement}%</strong></div>
                <div>Best: <strong>${mp.max_matches}/6</strong></div>
            </div></div>`;
    }
    
    html += '</div>';
    
    // Evidence
    if (v.evidence) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📝</span> Tổng Kết Bằng Chứng Phase 2</div></div>
        <div style="font-family:monospace;font-size:0.8rem;line-height:1.8;">`;
        v.evidence.forEach(e => {
            html += `<div style="color:${e.startsWith('+')?'#ef4444':'#22c55e'};">${e}</div>`;
        });
        html += '</div></div>';
    }
    
    container.innerHTML = html;
}

// ---- Temporal & Financial Analysis ----
function runTemporal(type) {
    const btn = document.getElementById('btn-temporal-' + type);
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '⏳ Đang phân tích thời gian & dòng tiền...';
    btn.classList.add('pulsing');
    
    const container = document.getElementById('temporal-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card" style="border-color: var(--accent-6);"><div class="loading">
        <div class="spinner"></div>
        <span class="loading-text">📅 Đang phân tích ngày tháng, jackpot, chu kỳ mặt trăng...<br>
        10 phương pháp temporal & financial analysis</span>
    </div></div>`;
    
    fetch(`/api/temporal/${type}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }
    })
    .then(r => r.json())
    .then(data => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        btn.classList.remove('pulsing');
        if (data.error) {
            container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`;
            return;
        }
        renderTemporal(container, data, type);
        showToast('Phân tích Temporal hoàn tất! 📅', 'success');
    })
    .catch(err => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        btn.classList.remove('pulsing');
        container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">❌</div><p>${err.message}</p></div></div>`;
    });
}

function renderTemporal(container, data, type) {
    let html = '';
    const v = data.verdict || {};
    const score = v.score || 0;
    const sColor = score >= 60 ? '#22c55e' : score >= 30 ? '#eab308' : '#3b82f6';
    
    // Verdict
    html += `<div class="card" style="border-color: ${sColor}; box-shadow: 0 0 30px ${sColor}33;">
        <div class="card-header"><div class="card-title"><span class="icon">📅</span> PHÂN TÍCH THỜI GIAN & DÒNG TIỀN - KẾT QUẢ</div></div>
        <div style="text-align:center; margin:16px 0;">
            <div style="font-size:2.5rem; font-weight:900; color:${sColor};">${score}%</div>
            <div style="font-size:1rem; font-weight:700; color:${sColor}; margin-top:6px;">"${v.verdict || ''}"</div>
            <div style="font-size:0.85rem; color:var(--text-muted);">${v.pattern_count || 0}/${v.total_tests || 0} phát hiện pattern</div>
        </div>
    </div>`;
    
    // Summary table
    const tests = ['day_of_week', 'monthly', 'jackpot_correlation', 'sequence_pattern', 
                   'date_number_corr', 'lunar', 'fibonacci_prime', 'year_comparison', 
                   'sum_by_date', 'temporal_prediction'];
    const icons = {'day_of_week':'📆','monthly':'🗓️','jackpot_correlation':'💰','sequence_pattern':'🔢',
                   'date_number_corr':'🔗','lunar':'🌙','fibonacci_prime':'🔟','year_comparison':'📊',
                   'sum_by_date':'📈','temporal_prediction':'🎯'};
    
    html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📋</span> 10 Phân Tích Chi Tiết</div></div>
    <div class="data-table-wrap"><table class="data-table"><thead><tr>
        <th></th><th>Phân Tích</th><th>Kết Quả</th><th>Kết Luận</th>
    </tr></thead><tbody>`;
    
    tests.forEach(key => {
        const t = data[key];
        if (!t) return;
        const isPat = t.is_pattern;
        const st = isPat ? '🔴 PATTERN' : '🟢 RANDOM';
        const rs = isPat ? 'background:rgba(239,68,68,0.1);' : '';
        html += `<tr style="${rs}"><td>${icons[key]||'📊'}</td><td><strong>${t.name||key}</strong></td>
            <td><span style="padding:3px 10px;border-radius:12px;font-weight:700;font-size:0.8rem;
            background:${isPat?'rgba(239,68,68,0.2)':'rgba(34,197,94,0.2)'};
            color:${isPat?'#ef4444':'#22c55e'};">${st}</span></td>
            <td style="font-size:0.8rem;color:var(--text-muted);">${t.conclusion||''}</td></tr>`;
    });
    html += '</tbody></table></div></div>';
    
    // Day of Week detail
    if (data.day_of_week && data.day_of_week.day_stats) {
        const ds = data.day_of_week.day_stats;
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📆</span> Số Nóng Theo Ngày Trong Tuần</div></div>
        <div class="data-table-wrap"><table class="data-table"><thead><tr>
            <th>Ngày</th><th>Số Kỳ</th><th>TB Tổng</th><th>Top 5 Số Nóng</th>
        </tr></thead><tbody>`;
        for (const [day, s] of Object.entries(ds)) {
            html += `<tr><td><strong>${day}</strong></td><td>${s.count}</td>
                <td>${s.avg_sum}</td><td>${s.top_5_numbers ? s.top_5_numbers.map(([n,c]) => 
                `<span class="ball small">${n.toString().padStart(2,'0')}</span>`).join('') : '-'}</td></tr>`;
        }
        html += '</tbody></table></div></div>';
    }
    
    // Jackpot correlation
    if (data.jackpot_correlation && data.jackpot_correlation.group_comparison) {
        const jc = data.jackpot_correlation;
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">💰</span> Ảnh Hưởng Jackpot Đến Kết Quả</div></div>
        <div class="grid-2">
            <div class="stat-card"><div class="stat-value">${jc.sum_jackpot_correlation}</div><div class="stat-label">Tương quan Tổng-Jackpot</div></div>
            <div class="stat-card"><div class="stat-value">${jc.jackpot_range ? (jc.jackpot_range.max/1e9).toFixed(0)+'B' : '-'}</div><div class="stat-label">Jackpot Tối Đa (VNĐ)</div></div>
        </div>`;
        if (jc.group_comparison) {
            html += '<div style="margin-top:12px;"><strong>Top Số theo mức Jackpot:</strong><div class="grid-2" style="margin-top:8px;">';
            for (const [group, gs] of Object.entries(jc.group_comparison)) {
                html += `<div class="prediction-card"><div class="model-name">💵 ${group.toUpperCase()} Jackpot</div>
                    <div>${gs.top_5 ? gs.top_5.map(([n]) => `<span class="ball small">${n.toString().padStart(2,'0')}</span>`).join('') : '-'}</div></div>`;
            }
            html += '</div></div>';
        }
        html += '</div>';
    }
    
    // Lunar phase
    if (data.lunar && data.lunar.phase_stats) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🌙</span> Phân Tích Theo Pha Mặt Trăng</div></div>
        <div class="grid-2">`;
        for (const [phase, ps] of Object.entries(data.lunar.phase_stats)) {
            html += `<div class="prediction-card"><div class="model-name">${phase === 'Full Moon' ? '🌕' : phase === 'New Moon' ? '🌑' : phase === 'First Quarter' ? '🌓' : '🌗'} ${phase}</div>
                <div>TB Tổng: <strong>${ps.avg_sum}</strong> (${ps.draws} kỳ)</div>
                <div style="margin-top:6px;">${ps.top_5 ? ps.top_5.map(([n]) => `<span class="ball small">${n.toString().padStart(2,'0')}</span>`).join('') : ''}</div></div>`;
        }
        html += '</div></div>';
    }
    
    // Temporal Predictor
    if (data.temporal_prediction) {
        const tp = data.temporal_prediction;
        html += `<div class="card" style="border-color: var(--accent-4);"><div class="card-header"><div class="card-title"><span class="icon">🎯</span> Temporal Predictor - Kết Quả Backtest</div></div>
        <div class="grid-4">
            <div class="stat-card"><div class="stat-value">${tp.avg_matches || '-'}</div><div class="stat-label">TB Đúng/6</div></div>
            <div class="stat-card"><div class="stat-value">${tp.random_expected || '-'}</div><div class="stat-label">Random Expected</div></div>
            <div class="stat-card"><div class="stat-value" style="color:${(tp.improvement||0) > 10 ? '#22c55e' : '#ef4444'};">${tp.improvement > 0 ? '+' : ''}${tp.improvement || 0}%</div><div class="stat-label">So Với Random</div></div>
            <div class="stat-card"><div class="stat-value">${tp.max_matches || '-'}/6</div><div class="stat-label">Tốt Nhất</div></div>
        </div></div>`;
    }
    
    // Evidence
    if (v.evidence) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📝</span> Tổng Kết Bằng Chứng</div></div>
        <div style="font-family:monospace;font-size:0.8rem;line-height:1.8;">`;
        v.evidence.forEach(e => {
            html += `<div style="color:${e.startsWith('+') ? '#ef4444' : '#22c55e'};">${e}</div>`;
        });
        html += '</div></div>';
    }
    
    container.innerHTML = html;
}

// ---- PRNG Cracker ----
function runCracker(type) {
    const btn = document.getElementById('btn-crack-' + type);
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '⏳ Đang phá giải thuật toán...';
    btn.classList.add('pulsing');
    
    const container = document.getElementById('crack-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card" style="border-color: #dc2626;"><div class="loading">
        <div class="spinner"></div>
        <span class="loading-text">🔓 Đang chạy 10 phương pháp khoa học phá giải PRNG...<br>
        LCG Detection, FFT Spectral, Autocorrelation, Runs Test,<br>
        Serial Correlation, Entropy, KS Test, Neural Cracker...</span>
    </div></div>`;
    
    fetch(`/api/crack/${type}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(r => r.json())
    .then(data => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        btn.classList.remove('pulsing');
        if (data.error) {
            container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`;
            return;
        }
        renderCracker(container, data, type);
        showToast('Phân tích PRNG hoàn tất! 🔓', 'success');
    })
    .catch(err => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        btn.classList.remove('pulsing');
        container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">❌</div><p>${err.message}</p></div></div>`;
    });
}

function renderCracker(container, data, type) {
    let html = '';
    const v = data.verdict || {};
    const score = v.prng_score || 0;
    const scoreColor = score >= 60 ? '#22c55e' : score >= 30 ? '#eab308' : '#dc2626';
    const scoreLabel = score >= 60 ? 'PHÁT HIỆN PRNG!' : score >= 30 ? 'KHÔNG RÕ RÀNG' : 'NGẪU NHIÊN THẬT';
    
    // Verdict Header
    html += `<div class="card" style="border-color: ${scoreColor}; box-shadow: 0 0 40px ${scoreColor}33;">
        <div class="card-header"><div class="card-title"><span class="icon">🔬</span> KẾT QUẢ PHÁ GIẢI - 10 PHƯƠNG PHÁP KHOA HỌC</div></div>
        <div style="text-align:center; margin: 20px 0;">
            <div style="font-size: 3rem; font-weight: 900; color: ${scoreColor};">${score}%</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: ${scoreColor}; margin-top: 8px;">${scoreLabel}</div>
            <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 8px;">
                ${v.prng_indicators || 0}/${v.total_tests || 0} phương pháp phát hiện pattern
            </div>
        </div>
        <div style="font-size: 0.9rem; line-height: 1.6; padding: 16px; background: rgba(0,0,0,0.3); border-radius: 8px;">
            <strong>"${v.verdict || ''}"</strong>
        </div>
    </div>`;
    
    // Individual Test Results
    const testOrder = ['lcg', 'fft', 'autocorrelation', 'runs_test', 'serial_correlation', 
                       'entropy', 'ks_test', 'modular', 'difference', 'neural'];
    const testIcons = {
        'lcg': '🔢', 'fft': '📡', 'autocorrelation': '📈', 'runs_test': '🏃',
        'serial_correlation': '🔗', 'entropy': '🎲', 'ks_test': '📏',
        'modular': '➗', 'difference': '📐', 'neural': '🧠'
    };
    
    html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📋</span> Chi Tiết 10 Phương Pháp Phân Tích</div></div>`;
    html += `<div class="data-table-wrap"><table class="data-table"><thead><tr>
        <th></th><th>Phương Pháp</th><th>Kết Quả</th><th>Kết Luận</th>
    </tr></thead><tbody>`;
    
    testOrder.forEach(key => {
        const test = data[key];
        if (!test) return;
        const icon = testIcons[key] || '📊';
        const isPrng = test.is_prng;
        const status = isPrng ? '🔴 PATTERN' : '🟢 RANDOM';
        const rowStyle = isPrng ? 'background: rgba(239,68,68,0.1);' : '';
        
        html += `<tr style="${rowStyle}">
            <td style="font-size:1.3rem;">${icon}</td>
            <td><strong>${test.name || key}</strong></td>
            <td><span style="padding:3px 12px; border-radius:12px; font-weight:700; font-size:0.8rem;
                background: ${isPrng ? 'rgba(239,68,68,0.2)' : 'rgba(34,197,94,0.2)'}; 
                color: ${isPrng ? '#ef4444' : '#22c55e'};">${status}</span></td>
            <td style="font-size:0.8rem; color:var(--text-muted);">${test.conclusion || ''}</td>
        </tr>`;
    });
    html += '</tbody></table></div></div>';
    
    // Detailed Results for each test
    html += '<div class="grid-2">';
    
    // LCG
    if (data.lcg) {
        const lcg = data.lcg;
        html += `<div class="prediction-card">
            <div class="model-name">🔢 LCG Detection</div>
            <p style="font-size:0.8rem; color:var(--text-muted);">Thử tìm công thức: X(n+1) = (a×X(n) + b) mod m</p>`;
        if (lcg.best_params) {
            html += `<div style="font-family:monospace; background:rgba(0,0,0,0.3); padding:10px; border-radius:6px; margin-top:8px;">
                <div>Formula: <strong>${lcg.best_params.formula}</strong></div>
                <div>Accuracy: <strong style="color:${lcg.confidence > 50 ? '#22c55e' : '#ef4444'};">${lcg.confidence}%</strong></div>
            </div>`;
        } else {
            html += `<div style="color:var(--text-muted);">Không tìm thấy LCG parameters</div>`;
        }
        html += '</div>';
    }
    
    // FFT
    if (data.fft) {
        html += `<div class="prediction-card">
            <div class="model-name">📡 FFT Spectral</div>
            <p style="font-size:0.8rem; color:var(--text-muted);">Phân tích Fourier tìm tần số ẩn</p>
            <div style="margin-top:8px;">Peaks: <strong>${data.fft.num_significant_peaks}</strong> | Max σ: <strong>${data.fft.max_sigma}</strong></div>`;
        if (data.fft.peaks && data.fft.peaks.length > 0) {
            html += '<div style="margin-top:8px; font-size:0.75rem;">';
            data.fft.peaks.slice(0, 5).forEach(p => {
                html += `<div>Period: ${p.period} | Magnitude: ${p.magnitude} | ${p.sigma}σ</div>`;
            });
            html += '</div>';
        }
        html += '</div>';
    }
    
    // Entropy
    if (data.entropy) {
        const e = data.entropy;
        const ratio = e.entropy_ratio;
        html += `<div class="prediction-card">
            <div class="model-name">🎲 Entropy Analysis</div>
            <p style="font-size:0.8rem; color:var(--text-muted);">Shannon entropy (1.0 = perfect random)</p>
            <div style="margin-top:8px;">
                <div>Entropy: <strong>${e.shannon_entropy}</strong> / ${e.max_entropy}</div>
                <div>Ratio: <strong style="color:${ratio > 0.98 ? '#22c55e' : '#eab308'};">${ratio}</strong></div>
                <div>Bigram Ratio: <strong>${e.bigram_ratio}</strong></div>
                <div style="margin-top:6px; background:rgba(0,0,0,0.3); border-radius:4px; height:12px; overflow:hidden;">
                    <div style="width:${ratio*100}%; height:100%; background: ${ratio > 0.98 ? '#22c55e' : '#eab308'};"></div>
                </div>
            </div>
        </div>`;
    }
    
    // Neural
    if (data.neural) {
        const n = data.neural;
        html += `<div class="prediction-card">
            <div class="model-name">🧠 Neural Cracker</div>
            <p style="font-size:0.8rem; color:var(--text-muted);">Neural network cố gắng học thuật toán ẩn</p>
            <div style="margin-top:8px;">
                <div>Avg matches: <strong>${n.avg_matches}</strong>/6</div>
                <div>Random expected: <strong>${n.random_expected}</strong>/6</div>
                <div>Improvement: <strong style="color:${n.improvement_over_random > 20 ? '#22c55e' : '#ef4444'};">
                    ${n.improvement_over_random > 0 ? '+' : ''}${n.improvement_over_random}%</strong></div>
                <div>Loss reduction: ${n.loss_reduction}%</div>
            </div>
        </div>`;
    }
    
    // KS Test
    if (data.ks_test) {
        const ks = data.ks_test;
        html += `<div class="prediction-card">
            <div class="model-name">📏 Kolmogorov-Smirnov</div>
            <p style="font-size:0.8rem; color:var(--text-muted);">So sánh phân bố với uniform</p>
            <div style="margin-top:8px;">
                <div>KS Statistic: <strong>${ks.ks_statistic}</strong></div>
                <div>Critical (95%): ${ks.critical_95}</div>
                <div>Pass 95%: <strong style="color:${ks.passes_95 ? '#22c55e' : '#ef4444'};">${ks.passes_95 ? 'YES' : 'NO'}</strong></div>
            </div>
        </div>`;
    }
    
    // Runs Test
    if (data.runs_test) {
        html += `<div class="prediction-card">
            <div class="model-name">🏃 Runs Test</div>
            <p style="font-size:0.8rem; color:var(--text-muted);">Wald-Wolfowitz randomness test</p>
            <div style="margin-top:8px;">
                <div>All pass: <strong style="color:${data.runs_test.all_pass ? '#22c55e' : '#ef4444'};">
                    ${data.runs_test.all_pass ? 'YES' : 'NO'}</strong></div>
                <div>Avg |Z|: ${data.runs_test.avg_abs_z_score}</div>
            </div>
        </div>`;
    }
    
    html += '</div>';
    
    // Evidence list
    if (v.evidence) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📝</span> Tổng Kết Bằng Chứng</div></div>
        <div style="font-family: monospace; font-size: 0.8rem; line-height: 1.8;">`;
        v.evidence.forEach(e => {
            const color = e.startsWith('+') ? '#ef4444' : '#22c55e';
            html += `<div style="color: ${color};">${e}</div>`;
        });
        html += '</div></div>';
    }
    
    container.innerHTML = html;
}

// ---- Backtest ----
function runBacktest(type) {
    const btn = document.getElementById('btn-backtest-' + type);
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '⏳ Đang kiểm tra 12 mô hình...';
    btn.classList.add('pulsing');
    
    const container = document.getElementById('backtest-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card" style="border-color: var(--accent-4);"><div class="loading">
        <div class="spinner"></div>
        <span class="loading-text">Đang chạy Walk-Forward Backtest trên toàn bộ dữ liệu lịch sử...<br>
        Kiểm tra 12 mô hình dự đoán, mỗi mô hình chạy ~200 lần.<br>
        Vui lòng chờ 30-60 giây...</span>
    </div></div>`;
    
    fetch(`/api/backtest/${type}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_tests: 300 })
    })
    .then(r => r.json())
    .then(data => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        btn.classList.remove('pulsing');
        
        if (data.error) {
            container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`;
            return;
        }
        renderBacktest(container, data, type);
        showToast('Backtest hoàn tất! Xem kết quả bên dưới.', 'success');
    })
    .catch(err => {
        btn.disabled = false;
        btn.innerHTML = originalText;
        btn.classList.remove('pulsing');
        container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">❌</div><p>Lỗi: ${err.message}</p></div></div>`;
    });
}

function renderBacktest(container, data, type) {
    let html = '';
    const models = data.models || [];
    const totalIter = data.total_iterations || 0;
    
    // Header
    html += `<div class="card" style="border-color: var(--accent-4); box-shadow: 0 0 30px rgba(245,158,11,0.2);">
        <div class="card-header"><div class="card-title"><span class="icon">🧪</span> KẾT QUẢ BACKTEST - Kiểm Tra Dự Đoán vs Thực Tế</div></div>
        <p style="color: var(--text-muted); margin-bottom: 16px;">
            Đã kiểm tra <strong style="color: var(--accent-4);">${totalIter} lần</strong> trên dữ liệu lịch sử thực. 
            Mỗi lần: dùng dữ liệu trước đó để dự đoán kỳ tiếp, rồi so sánh với kết quả thật.
        </p>`;
    
    // Leaderboard table
    html += `<div class="data-table-wrap"><table class="data-table"><thead><tr>
        <th>Rank</th><th>Mô Hình</th><th>Trung Bình Đúng</th><th>Tốt Nhất</th>
        <th>≥3 Đúng</th><th>≥2 Đúng</th><th>Phân Bố Kết Quả</th>
    </tr></thead><tbody>`;
    
    models.forEach((m, idx) => {
        const medal = idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `#${idx+1}`;
        const rowStyle = idx < 3 ? 'background: rgba(245,158,11,0.05);' : '';
        
        // Distribution bars
        let distHtml = '<div style="display:flex; gap:2px; align-items:end; height:30px;">';
        for (let i = 0; i <= 6; i++) {
            const count = m.match_distribution[String(i)] || 0;
            const pct = totalIter > 0 ? (count / totalIter * 100) : 0;
            const height = Math.max(2, pct * 0.6);
            const color = i >= 4 ? '#22c55e' : i >= 3 ? '#eab308' : i >= 2 ? '#3b82f6' : '#6b7280';
            distHtml += `<div title="${i} đúng: ${count}x (${pct.toFixed(1)}%)" 
                style="width:12px; height:${height}px; background:${color}; border-radius:2px 2px 0 0;"></div>`;
        }
        distHtml += '</div>';
        
        html += `<tr style="${rowStyle}">
            <td style="font-size:1.2rem; text-align:center;">${medal}</td>
            <td><strong>${m.model}</strong></td>
            <td style="font-weight:700; color: ${m.avg_matches >= 1.0 ? 'var(--accent-4)' : 'var(--text-primary)'};">
                ${m.avg_matches.toFixed(3)} / 6
            </td>
            <td><span style="background: ${m.best_score >= 4 ? 'var(--gradient-gold)' : m.best_score >= 3 ? 'var(--gradient-1)' : 'var(--gradient-2)'}; 
                padding: 2px 10px; border-radius: 12px; font-weight: 700;">${m.best_score}/6</span></td>
            <td>${m.at_least_3}%</td>
            <td>${m.at_least_2}%</td>
            <td>${distHtml}</td>
        </tr>`;
    });
    html += '</tbody></table></div>';
    
    // Distribution legend
    html += `<div style="display:flex; gap:16px; margin-top:12px; font-size:0.75rem; color:var(--text-muted);">
        <span><span style="display:inline-block;width:10px;height:10px;background:#6b7280;border-radius:2px;"></span> 0-1 đúng</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#3b82f6;border-radius:2px;"></span> 2 đúng</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#eab308;border-radius:2px;"></span> 3 đúng</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#22c55e;border-radius:2px;"></span> 4+ đúng</span>
    </div>`;
    html += '</div>';
    
    // Best single prediction
    if (models.length > 0 && models[0].best_detail) {
        const best = models[0].best_detail;
        html += `<div class="card" style="border-color: #22c55e;">
            <div class="card-header"><div class="card-title"><span class="icon">🏆</span> Lần Dự Đoán Tốt Nhất</div></div>
            <div style="margin-bottom:10px;">Model: <strong>${models[0].model}</strong> | Kỳ quay #${best.draw_index} | Đúng: <strong style="color:#22c55e;">${best.matches}/6</strong></div>
            <div style="display:flex; gap:24px; flex-wrap:wrap;">
                <div><div style="color:var(--text-muted); margin-bottom:6px;">Dự đoán:</div>
                    <div class="ball-container">${best.predicted.map(n => `<span class="ball small">${n.toString().padStart(2,'0')}</span>`).join('')}</div>
                </div>
                <div><div style="color:var(--text-muted); margin-bottom:6px;">Thực tế:</div>
                    <div class="ball-container">${best.actual.map(n => {
                        const isMatch = best.predicted.includes(n);
                        return `<span class="ball small" style="${isMatch ? 'background:var(--gradient-gold);box-shadow:0 0 10px rgba(245,158,11,0.5);' : ''}">${n.toString().padStart(2,'0')}</span>`;
                    }).join('')}</div>
                </div>
            </div>
        </div>`;
    }
    
    // Recent predictions details for top model
    if (models.length > 0 && models[0].recent_results && models[0].recent_results.length > 0) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📋</span> Chi Tiết 10 Lần Test Gần Nhất (${models[0].model})</div></div>`;
        html += `<div class="data-table-wrap"><table class="data-table"><thead><tr>
            <th>Kỳ</th><th>Dự Đoán</th><th>Thực Tế</th><th>Đúng</th>
        </tr></thead><tbody>`;
        
        models[0].recent_results.forEach(r => {
            const matchNums = new Set(r.predicted.filter(n => r.actual.includes(n)));
            html += `<tr>
                <td>#${r.draw_index}</td>
                <td><div class="ball-container">${r.predicted.map(n => 
                    `<span class="ball small" style="${matchNums.has(n) ? 'background:var(--gradient-gold);' : ''}">${n.toString().padStart(2,'0')}</span>`
                ).join('')}</div></td>
                <td><div class="ball-container">${r.actual.map(n => 
                    `<span class="ball small" style="${matchNums.has(n) ? 'background:var(--gradient-gold);' : ''}">${n.toString().padStart(2,'0')}</span>`
                ).join('')}</div></td>
                <td style="font-weight:700; color:${r.matches >= 3 ? '#22c55e' : r.matches >= 2 ? '#eab308' : 'var(--text-muted)'};">${r.matches}/6</td>
            </tr>`;
        });
        html += '</tbody></table></div></div>';
    }
    
    // Conclusion card
    const bestModel = models.length > 0 ? models[0] : null;
    const avgBest = bestModel ? bestModel.avg_matches : 0;
    
    html += `<div class="card" style="border-color: var(--accent-3);">
        <div class="card-header"><div class="card-title"><span class="icon">📊</span> Kết Luận</div></div>
        <div style="font-size: 0.95rem; line-height: 1.8;">
            <p>Sau <strong>${totalIter}</strong> lần kiểm tra trên dữ liệu thực:</p>
            <ul style="margin-top:8px; padding-left:20px;">
                <li>Mô hình tốt nhất: <strong style="color:var(--accent-4);">${bestModel ? bestModel.model : 'N/A'}</strong> với trung bình <strong>${avgBest.toFixed(3)}/6</strong> số đúng</li>
                <li>Tỷ lệ ≥3 đúng: <strong>${bestModel ? bestModel.at_least_3 : 0}%</strong></li>
                <li>Lần tốt nhất: <strong style="color:#22c55e;">${bestModel ? bestModel.best_score : 0}/6</strong></li>
                <li>Xác suất ngẫu nhiên (Random): ~<strong>${models.find(m => m.model === 'Random Baseline') ? models.find(m => m.model === 'Random Baseline').avg_matches.toFixed(3) : '0.8'}/6</strong></li>
            </ul>
            <p style="margin-top:12px; color:var(--accent-3); font-style:italic;">
                ⚠️ Kết quả cho thấy tất cả các mô hình đều có kết quả tương tự Random Baseline, 
                chứng minh xổ số thực sự ngẫu nhiên và không thể dự đoán 100%.
            </p>
        </div>
    </div>`;
    
    container.innerHTML = html;
}

// ---- Advanced Analysis ----
function loadAdvanced(type) {
    const container = document.getElementById('advanced-' + type);
    container.style.display = 'block';
    container.innerHTML = `<div class="card"><div class="loading"><div class="spinner"></div><span class="loading-text">Đang chạy 8 phương pháp phân tích chuyên sâu...</span></div></div>`;
    
    fetch(`/api/advanced/${type}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">⚠️</div><p>${data.error}</p></div></div>`;
                return;
            }
            renderAdvanced(container, data.advanced, type);
            showToast('Phân tích chuyên sâu hoàn tất! ⚡', 'success');
        })
        .catch(err => {
            container.innerHTML = `<div class="card"><div class="empty-state"><div class="icon">❌</div><p>Lỗi: ${err.message}</p></div></div>`;
        });
}

function renderAdvanced(container, adv, type) {
    let html = '';
    
    // === TOP SCORED NUMBERS (Composite) ===
    html += `<div class="card" style="border-color: var(--accent-4); box-shadow: 0 0 30px rgba(245,158,11,0.15);">
        <div class="card-header"><div class="card-title"><span class="icon">🏆</span> TOP SỐ ĐIỂM CAO NHẤT (Tổng hợp 8 phương pháp)</div></div>`;
    
    if (adv.top_numbers) {
        html += '<div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;">';
        adv.top_numbers.forEach((item, idx) => {
            const n = typeof item.number === 'string' ? parseInt(item.number) : item.number;
            const gradient = idx < 3 ? 'var(--gradient-gold)' : idx < 7 ? 'var(--gradient-1)' : 'var(--gradient-2)';
            html += `<div style="text-align:center;">
                <span class="ball" style="background:${gradient}; width:58px; height:58px; font-size:1.2rem;">${n.toString().padStart(2,'0')}</span>
                <div style="font-size:0.75rem; color:var(--accent-4); margin-top:6px; font-weight:700;">${item.score.toFixed(1)} pts</div>
                <div style="font-size:0.65rem; color:var(--text-muted);">#${idx+1}</div>
            </div>`;
        });
        html += '</div>';
    }
    html += '</div>';
    
    // === 8 METHOD BREAKDOWN ===
    html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🔬</span> Chi Tiết 8 Phương Pháp Phân Tích</div></div>`;
    
    if (adv.methods) {
        const methodIcons = {
            'markov': '🔗', 'bayesian': '📐', 'association': '🤝',
            'gap_cycle': '⏰', 'chi_square': '📏', 'monte_carlo': '🎲',
            'patterns': '🧩', 'recency': '📅'
        };
        const methodDescriptions = {
            'markov': 'Xác suất chuyển tiếp: số nào hay xuất hiện SAU kỳ trước',
            'bayesian': 'Cập nhật xác suất Bayesian theo thời gian, ưu tiên gần đây',
            'association': 'Tìm các cặp số hay đi cùng nhau (Lift & Confidence)',
            'gap_cycle': 'Phân tích chu kỳ: số nào đã "quá hạn" cần xuất hiện',
            'chi_square': 'Phát hiện số lệch khỏi phân bố đều (cần cân bằng)',
            'monte_carlo': 'Mô phỏng 100,000 lần quay để ước lượng xác suất',
            'patterns': 'Phân tích cấu trúc: Chẵn/Lẻ, Cao/Thấp, Tổng, Liên tiếp',
            'recency': 'Phân tích có trọng số: kỳ gần đây quan trọng hơn'
        };
        
        html += '<div class="grid-2">';
        for (const [key, info] of Object.entries(adv.methods)) {
            const icon = methodIcons[key] || '📊';
            const desc = methodDescriptions[key] || '';
            
            html += `<div class="prediction-card">
                <div class="model-name">${icon} ${info.name}</div>
                <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:10px;">${desc}</div>
                <div style="display:flex; gap:6px; flex-wrap:wrap;">`;
            
            if (info.top_5) {
                info.top_5.forEach(([num, score]) => {
                    const n = typeof num === 'string' ? parseInt(num) : num;
                    html += `<div style="text-align:center;">
                        <span class="ball small">${n.toString().padStart(2,'0')}</span>
                        <div style="font-size:0.65rem; color:var(--text-muted);">${score.toFixed(0)}</div>
                    </div>`;
                });
            }
            html += '</div></div>';
        }
        html += '</div>';
    }
    html += '</div>';
    
    // === PATTERN INFO ===
    if (adv.pattern_info) {
        const p = adv.pattern_info;
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🧩</span> Phân Tích Cấu Trúc Tối Ưu</div></div>
            <div class="grid-4">
                <div class="stat-card"><div class="stat-value">${p.optimal_odd || '-'}/${6 - (p.optimal_odd || 0)}</div><div class="stat-label">Lẻ / Chẵn Tối Ưu</div></div>
                <div class="stat-card"><div class="stat-value">${p.optimal_high || '-'}/${6 - (p.optimal_high || 0)}</div><div class="stat-label">Cao / Thấp Tối Ưu</div></div>
                <div class="stat-card"><div class="stat-value">${p.avg_sum || '-'}</div><div class="stat-label">Tổng Trung Bình</div></div>
                <div class="stat-card"><div class="stat-value">${p.sum_range ? p.sum_range[0]+'-'+p.sum_range[1] : '-'}</div><div class="stat-label">Khoảng Tổng 68%</div></div>
            </div>`;
        
        // Odd/Even distribution chart
        if (p.odd_even_dist) {
            html += '<div style="margin-top:16px;"><strong style="color:var(--accent-6);">📊 Phân bố Lẻ/Chẵn:</strong><div style="margin-top:8px;">';
            for (const [odd, count] of Object.entries(p.odd_even_dist)) {
                const pct = count; // Will show count
                html += `<div class="freq-bar-wrap">
                    <span class="freq-number">${odd}L/${6-odd}C</span>
                    <div class="freq-bar-bg"><div class="freq-bar" style="width:${Math.min(count * 0.3, 100)}%"></div></div>
                    <span class="freq-count">${count}x</span>
                </div>`;
            }
            html += '</div></div>';
        }
        html += '</div>';
    }
    
    // === GAP DETAILS (top numbers) ===
    if (adv.gap_details && Object.keys(adv.gap_details).length > 0) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">⏰</span> Chi Tiết Chu Kỳ (Top Số Điểm Cao)</div></div>
        <div class="data-table-wrap"><table class="data-table"><thead><tr>
            <th>Số</th><th>Lần XH</th><th>Gap TB</th><th>Gap Hiện Tại</th><th>Tỷ Lệ Quá Hạn</th><th>Trạng Thái</th>
        </tr></thead><tbody>`;
        
        for (const [num, info] of Object.entries(adv.gap_details)) {
            if (!info || !info.avg_gap) continue;
            const ratio = info.overdue_ratio || 0;
            const status = ratio > 1.3 ? '🔴 QUÁ HẠN' : ratio > 0.8 ? '🟡 SẮP ĐẾN' : '🟢 Bình thường';
            const rowStyle = ratio > 1.3 ? 'background: rgba(239,68,68,0.1);' : '';
            
            html += `<tr style="${rowStyle}">
                <td><span class="ball small">${parseInt(num).toString().padStart(2,'0')}</span></td>
                <td>${info.appearances || '-'}</td>
                <td>${info.avg_gap}</td>
                <td style="font-weight:700; color:${ratio > 1 ? 'var(--accent-3)' : 'var(--text-primary)'};">${info.current_gap}</td>
                <td><strong>${(ratio * 100).toFixed(0)}%</strong></td>
                <td>${status}</td>
            </tr>`;
        }
        html += '</tbody></table></div></div>';
    }
    
    // === TOP PAIRS (Association) ===
    if (adv.top_pairs && adv.top_pairs.length > 0) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">🤝</span> Cặp Số Hay Đi Cùng Nhau</div></div>
        <div style="display:flex; flex-wrap:wrap; gap:12px;">`;
        
        adv.top_pairs.forEach(pair => {
            html += `<div class="prediction-card" style="min-width:140px; text-align:center; flex:0 0 auto;">
                <div class="ball-container" style="justify-content:center;">
                    <span class="ball small">${pair.pair[0].toString().padStart(2,'0')}</span>
                    <span style="color:var(--text-muted);">+</span>
                    <span class="ball small">${pair.pair[1].toString().padStart(2,'0')}</span>
                </div>
                <div style="font-size:0.75rem; color:var(--accent-4); margin-top:6px;">${pair.count}x (${pair.support}%)</div>
            </div>`;
        });
        html += '</div></div>';
    }
    
    // === CHI-SQUARE ===
    if (adv.chi_square) {
        html += `<div class="card"><div class="card-header"><div class="card-title"><span class="icon">📏</span> Kiểm Định Chi-Square</div></div>
            <div class="grid-2">
                <div class="stat-card"><div class="stat-value">${adv.chi_square.total || '-'}</div><div class="stat-label">Tổng Chi-Square (χ²)</div></div>
                <div class="stat-card"><div class="stat-value">${adv.chi_square.df || '-'}</div><div class="stat-label">Bậc Tự Do (df)</div></div>
            </div>
        </div>`;
    }
    
    container.innerHTML = html;
}

// ---- Scrape Data ----
function scrapeData() {
    if (!confirm('Thu thập toàn bộ dữ liệu từ ketquadientoan.com?\nQuá trình này có thể mất 1-3 phút.')) return;
    
    showToast('🔄 Đang thu thập dữ liệu... Vui lòng chờ.', 'info');
    
    fetch('/api/scrape', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                showToast('❌ Lỗi: ' + data.error, 'error');
                return;
            }
            document.getElementById('mega-count').textContent = data.mega_count;
            document.getElementById('power-count').textContent = data.power_count;
            showToast(`✅ Thu thập thành công! Mega: ${data.mega_count} kỳ, Power: ${data.power_count} kỳ`, 'success');
            
            // Reload current tab data
            loadData(currentTab, 20);
        })
        .catch(err => {
            showToast('❌ Lỗi kết nối: ' + err.message, 'error');
        });
}

// ---- Toast Notification ----
function showToast(message, type = 'info') {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
