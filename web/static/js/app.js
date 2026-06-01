/**
 * 智能图表 - 前端主逻辑
 */

// ==================== 全局状态 ====================
let currentStep = 1;
let uploadedData = null;
let trainTimer = null;
let trainEventSinceId = -1;
let selectedOverrideModel = null;
let llmTimer = null;
let selectedAnalysisType = null;

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initPredictUpload();
    initModeHint();
    initDragDrop();
    loadDatasets();  // 加载已有数据集列表
});

// ==================== 步骤导航 ====================
function goStep(n) {
    currentStep = n;
    document.querySelectorAll('.step-item').forEach(el => el.classList.remove('active'));
    document.querySelector(`.step-item[data-step="${n}"]`).classList.add('active');
    document.querySelectorAll('.step-panel').forEach(el => el.classList.remove('active'));
    document.getElementById(`step-${n}`).classList.add('active');

    // 步骤特定加载
    if (n === 2 && uploadedData) loadEDA();
    if (n === 3 && uploadedData) initReportDesigner();
    if (n === 4 && uploadedData) loadAutoConfig();
    if (n === 7) initLLMStep();
}

function markStepCompleted(n) {
    document.querySelector(`.step-item[data-step="${n}"]`).classList.add('completed');
}

// ==================== 文件上传（支持多文件和多sheet） ====================
let uploadedFiles = [];  // 存储上传的文件信息
let selectedSheets = new Set();  // 合并时选中的sheet

function initUpload() {
    const input = document.getElementById('file-input');
    const area = document.getElementById('upload-area');

    input.addEventListener('change', e => {
        if (e.target.files.length) handleUpload(e.target.files);
    });

    area.addEventListener('click', e => {
        if (e.target.closest('label')) return;
        input.click();
    });
}

function initDragDrop() {
    const area = document.getElementById('upload-area');
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        area.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        area.addEventListener(eventName, () => area.classList.add('dragover'), false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        area.addEventListener(eventName, () => area.classList.remove('dragover'), false);
    });
    area.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length) handleUpload(files);
    }, false);
}

async function analyzeProblem() {
    const desc = document.getElementById('problem-description').value.trim();
    if (!desc) {
        showToast('请输入题目描述', 'error');
        return;
    }
    const resultDiv = document.getElementById('problem-result');
    const contentDiv = document.getElementById('problem-result-content');
    if (contentDiv) contentDiv.innerHTML = '<div class="hint">正在分析...</div>';
    if (resultDiv) resultDiv.style.display = '';
    try {
        const res = await fetch('/api/problem/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description: desc })
        });
        const data = await res.json();
        if (data.success && data.result) {
            const r = data.result;
            let html = '<div style="background:#f8f9fa;padding:12px;border-radius:6px;">';
            html += '<h4 style="margin:0 0 8px;color:#2c3e50;">' + r.model + ' <span style="font-size:12px;color:#27ae60;">(' + r.confidence + '% 匹配)</span></h4>';
            html += '<h5 style="margin:10px 0 4px;font-size:13px;">关键公式</h5><ul style="font-size:12px;margin:0;padding-left:16px;">';
            r.formulas.forEach(f => { html += '<li>' + f + '</li>'; });
            html += '</ul>';
            html += '<h5 style="margin:10px 0 4px;font-size:13px;">建模步骤</h5><ol style="font-size:12px;margin:0;padding-left:16px;">';
            r.approach.forEach(a => { html += '<li>' + a + '</li>'; });
            html += '</ol>';
            html += '<h5 style="margin:10px 0 4px;font-size:13px;">关键变量</h5><p style="font-size:12px;margin:0;">' + r.key_features.join('、') + '</p>';
            html += '<h5 style="margin:10px 0 4px;font-size:13px;">Python 代码框架</h5><pre style="background:#fff;padding:8px;border-radius:4px;font-size:11px;overflow:auto;border:1px solid #e5e7eb;">' + r.code_template.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</pre>';
            html += '</div>';
            if (contentDiv) contentDiv.innerHTML = html;
        } else {
            if (contentDiv) contentDiv.innerHTML = '<div class="hint">分析失败: ' + (data.error || '') + '</div>';
        }
    } catch (e) {
        if (contentDiv) contentDiv.innerHTML = '<div class="hint">错误: ' + e.message + '</div>';
    }
}

async function handleUpload(files) {
    const fileList = Array.from(files);
    const isAppend = uploadedFiles.length > 0;
    showToast(`正在上传 ${fileList.length} 个文件...`);
    const formData = new FormData();
    fileList.forEach(f => formData.append('files', f));

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.success) {
            uploadedData = data.data;
            uploadedFiles = data.files || [];
            selectedSheets.clear();
            showUploadResult(data);
            renderMultiTablePanel();
            const totalFiles = uploadedFiles.length;
            document.getElementById('data-status').textContent = `数据集: ${totalFiles} 个文件 · 当前 ${data.data.shape[0]}行×${data.data.shape[1]}列`;
            document.getElementById('data-status').classList.add('loaded');
            markStepCompleted(1);
            
            if (isAppend) {
                showToast(`追加成功！现有 ${totalFiles} 个数据集`);
            } else {
                showToast(`上传成功！共 ${totalFiles} 个文件`);
            }

            populateTargetOptions(data.data.columns, data.target_hint);
            goStep(2);
        } else {
            showToast(data.error || '上传失败', 'error');
        }
    } catch (e) {
        showToast('上传出错: ' + e.message, 'error');
    }
}

async function loadDatasets() {
    // 页面加载时获取已上传的数据集列表
    try {
        const res = await fetch('/api/datasets');
        const data = await res.json();
        if (data.success && data.datasets.length > 0) {
            uploadedFiles = data.datasets;
            renderMultiTablePanel();
            const active = data.datasets[data.active_index];
            if (active) {
                document.getElementById('data-status').textContent = `数据集: ${data.datasets.length} 个 · 当前 ${active.shape[0]}×${active.shape[1]}`;
                document.getElementById('data-status').classList.add('loaded');
                populateTargetOptions(active.columns, null);
                // 恢复 uploadedData，否则高级分析无法获取列类型
                if (!uploadedData) uploadedData = {};
                uploadedData.columns = active.columns;
                // 尝试从summary接口获取列类型
                try {
                    const sumRes = await fetch('/api/analytics/summary');
                    const sumData = await sumRes.json();
                    if (sumData.success && sumData.summary) {
                        uploadedData.column_types = {};
                        // 直接使用后端返回的分类，避免前端字符串推断出错
                        (sumData.summary.numeric_columns || []).forEach(c => uploadedData.column_types[c] = 'numeric');
                        (sumData.summary.categorical_columns || []).forEach(c => uploadedData.column_types[c] = 'categorical');
                        // datetime 列单独标记（根据 dtypes 字符串判断）
                        const dtypes = sumData.summary.dtypes || {};
                        Object.entries(dtypes).forEach(([col, dtype]) => {
                            const d = String(dtype).toLowerCase();
                            if (d.includes('datetime')) {
                                uploadedData.column_types[col] = 'datetime';
                            } else if (d.includes('bool') && uploadedData.column_types[col] !== 'categorical') {
                                uploadedData.column_types[col] = 'categorical';
                            }
                        });
                        // 缓存分布信息，用于卡方检验排除高基数列
                        uploadedData.distributions = sumData.summary.distributions || {};
                    }
                } catch (e2) {
                    // fallback: 简单推断
                    uploadedData.column_types = {};
                    active.columns.forEach(c => {
                        uploadedData.column_types[c] = 'numeric'; // 默认假设，让用户自己选择
                    });
                }
            }
        }
    } catch (e) {
        console.error('加载数据集列表失败', e);
    }
}

async function deleteDataset(index) {
    if (!confirm('确定要删除这个数据集吗？')) return;
    try {
        const res = await fetch(`/api/datasets/${index}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            uploadedFiles.splice(index, 1);
            renderMultiTablePanel();
            if (data.datasets_count > 0) {
                showToast('已删除，剩余 ' + data.datasets_count + ' 个数据集');
            } else {
                document.getElementById('data-status').textContent = '未上传数据';
                document.getElementById('data-status').classList.remove('loaded');
                document.getElementById('multi-table-panel').classList.add('hidden');
                document.getElementById('upload-result').classList.add('hidden');
                showToast('所有数据集已删除');
            }
        } else {
            showToast(data.error || '删除失败', 'error');
        }
    } catch (e) {
        showToast('删除出错: ' + e.message, 'error');
    }
}

function renderMultiTablePanel() {
    const panel = document.getElementById('multi-table-panel');
    const listEl = document.getElementById('file-sheet-list');
    panel.classList.remove('hidden');
    
    let html = '';
    uploadedFiles.forEach((file, fIdx) => {
        const hasSheets = file.sheets && file.sheets.length > 1;
        const icon = file.ext && file.ext.includes('xls') ? '📊' : '📄';
        const isActive = file.is_active;
        const activeStyle = isActive ? 'border-color:var(--primary);box-shadow:0 0 0 2px rgba(67,97,238,0.1);' : '';
        
        html += `
            <div class="file-group" style="margin-bottom:14px;padding:12px;border:1px solid var(--border);border-radius:8px;background:${isActive ? '#f0f7ff' : '#fafbfc'};${activeStyle}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <div style="font-weight:600;">${icon} ${escapeHtml(file.filename)} <span style="color:#999;font-size:12px;">${file.shape[0]}×${file.shape[1]}</span></div>
                    <div style="display:flex;gap:6px;align-items:center;">
                        ${isActive ? '<span style="font-size:11px;color:var(--primary);font-weight:600;">● 当前</span>' : ''}
                        <button class="btn btn-sm" style="padding:3px 8px;font-size:11px;color:#999;" onclick="deleteDataset(${fIdx})" title="删除">🗑️</button>
                    </div>
                </div>
        `;
        
        if (hasSheets) {
            html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">';
            file.sheets.forEach((sheet, sIdx) => {
                const sheetActive = isActive && file.active_sheet === sheet;
                const key = `${fIdx}::${sheet}`;
                html += `
                    <label class="sheet-tag ${sheetActive ? 'active' : ''}" data-key="${key}" style="cursor:pointer;padding:4px 10px;border-radius:4px;font-size:12px;border:1px solid var(--border);background:${sheetActive ? 'var(--primary)' : '#fff'};color:${sheetActive ? '#fff' : 'var(--text)'};">
                        <input type="checkbox" value="${key}" ${sheetActive ? 'checked' : ''} onchange="toggleSheetSelection(this)" style="display:none;">
                        ${escapeHtml(sheet)}
                    </label>
                `;
            });
            html += '</div>';
        } else {
            html += `<div style="font-size:12px;color:#999;">单表数据 · ${file.columns.length} 列</div>`;
        }
        
        if (!isActive) {
            html += `
                <div style="margin-top:8px;">
                    <button class="btn btn-sm" onclick="selectSheet(${fIdx}, '${escapeHtml(file.active_sheet || '')}')">📋 切换到此表</button>
                </div>
            `;
        }
        html += '</div>';
    });
    
    listEl.innerHTML = html;
    updateJoinOptions();
}

function toggleSheetSelection(checkbox) {
    const label = checkbox.closest('.sheet-tag');
    const key = checkbox.value;
    if (checkbox.checked) {
        selectedSheets.add(key);
        label.style.background = 'var(--primary)';
        label.style.color = '#fff';
        label.classList.add('active');
    } else {
        selectedSheets.delete(key);
        label.style.background = '#fff';
        label.style.color = 'var(--text)';
        label.classList.remove('active');
    }
}

async function selectSheet(fileIndex, sheetName) {
    try {
        const res = await fetch('/api/upload/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_index: fileIndex, sheet_name: sheetName })
        });
        const data = await res.json();
        if (data.success) {
            uploadedData = data.data;
            showUploadResult(data);
            
            // 更新本地状态
            uploadedFiles.forEach((f, i) => {
                f.is_active = (i === fileIndex);
                if (i === fileIndex) f.active_sheet = sheetName;
            });
            
            document.getElementById('data-status').textContent = `数据集: ${uploadedFiles.length} 个 · 当前 ${uploadedFiles[fileIndex].filename}${sheetName ? ' · ' + sheetName : ''} (${data.data.shape[0]}行×${data.data.shape[1]}列)`;
            populateTargetOptions(data.data.columns, data.target_hint);
            showToast('已切换数据表');
            
            // 重新渲染面板以更新活跃状态
            renderMultiTablePanel();
        } else {
            showToast(data.error || '切换失败', 'error');
        }
    } catch (e) {
        showToast('切换出错: ' + e.message, 'error');
    }
}

function onMultiOpChange() {
    const type = document.getElementById('multi-op-type').value;
    document.getElementById('merge-options').classList.toggle('hidden', type !== 'merge');
    document.getElementById('join-options').classList.toggle('hidden', type !== 'join');
}

async function executeMerge() {
    if (selectedSheets.size < 2) {
        showToast('请至少勾选2个sheet进行合并', 'error');
        return;
    }
    const axis = parseInt(document.getElementById('merge-axis').value);
    const sources = Array.from(selectedSheets).map(key => {
        const [fIdx, sheet] = key.split('::');
        return { file_index: parseInt(fIdx), sheet_name: sheet };
    });
    
    try {
        const res = await fetch('/api/upload/merge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sources, axis })
        });
        const data = await res.json();
        if (data.success) {
            uploadedData = data.data;
            showUploadResult(data);
            document.getElementById('data-status').textContent = `合并结果: ${data.shape[0]}行×${data.shape[1]}列`;
            populateTargetOptions(data.data.columns, data.target_hint);
            showToast(`合并成功！${data.shape[0]}行×${data.shape[1]}列`);
            goStep(2);
        } else {
            showToast(data.error || '合并失败', 'error');
        }
    } catch (e) {
        showToast('合并出错: ' + e.message, 'error');
    }
}

function updateJoinOptions() {
    const leftSel = document.getElementById('join-left');
    const rightSel = document.getElementById('join-right');
    if (!leftSel || !rightSel) return;
    
    let options = '<option value="">选择表...</option>';
    uploadedFiles.forEach((file, idx) => {
        if (file.sheets) {
            file.sheets.forEach(sheet => {
                const label = `${file.filename} · ${sheet}`;
                const value = JSON.stringify({ file_index: idx, sheet_name: sheet });
                options += `<option value='${escapeHtml(value)}'>${escapeHtml(label)}</option>`;
            });
        } else {
            const label = file.filename;
            const value = JSON.stringify({ file_index: idx, sheet_name: null });
            options += `<option value='${escapeHtml(value)}'>${escapeHtml(label)}</option>`;
        }
    });
    
    leftSel.innerHTML = options;
    rightSel.innerHTML = options;
    
    leftSel.onchange = updateJoinKeyOptions;
    rightSel.onchange = updateJoinKeyOptions;
}

async function updateJoinKeyOptions() {
    const leftVal = document.getElementById('join-left').value;
    const rightVal = document.getElementById('join-right').value;
    const onSel = document.getElementById('join-on');
    
    if (!leftVal || !rightVal) {
        onSel.innerHTML = '<option value="">先选择左右表</option>';
        return;
    }
    
    try {
        const left = JSON.parse(leftVal);
        const right = JSON.parse(rightVal);
        
        // 获取两个表的列名（从uploadedFiles中已有）
        const leftFile = uploadedFiles[left.file_index];
        const rightFile = uploadedFiles[right.file_index];
        const leftCols = new Set(leftFile.columns || []);
        const rightCols = new Set(rightFile.columns || []);
        const commonCols = [...leftCols].filter(c => rightCols.has(c));
        
        if (commonCols.length === 0) {
            onSel.innerHTML = '<option value="">两表无共同列</option>';
        } else {
            onSel.innerHTML = commonCols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
        }
    } catch (e) {
        onSel.innerHTML = '<option value="">选择错误</option>';
    }
}

async function executeJoin() {
    const leftVal = document.getElementById('join-left').value;
    const rightVal = document.getElementById('join-right').value;
    const on = document.getElementById('join-on').value;
    const how = document.getElementById('join-how').value;
    
    if (!leftVal || !rightVal || !on) {
        showToast('请完整选择左表、右表和关联键', 'error');
        return;
    }
    
    try {
        const res = await fetch('/api/upload/join', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                left: JSON.parse(leftVal),
                right: JSON.parse(rightVal),
                on: on,
                how: how
            })
        });
        const data = await res.json();
        if (data.success) {
            uploadedData = data.data;
            showUploadResult(data);
            document.getElementById('data-status').textContent = `关联结果: ${data.shape[0]}行×${data.shape[1]}列`;
            populateTargetOptions(data.data.columns, data.target_hint);
            showToast(`关联成功！${data.shape[0]}行×${data.shape[1]}列`);
            goStep(2);
        } else {
            showToast(data.error || '关联失败', 'error');
        }
    } catch (e) {
        showToast('关联出错: ' + e.message, 'error');
    }
}

function showUploadResult(data) {
    const preview = document.getElementById('upload-preview');
    const df = data.data;
    let html = `<p><strong>形状:</strong> ${df.shape[0]} 行 × ${df.shape[1]} 列 | <strong>内存:</strong> ${df.memory_mb} MB</p>`;
    html += '<div class="table-wrapper"><table class="data-table"><thead><tr>';
    df.columns.forEach(c => html += `<th>${c}</th>`);
    html += '</tr></thead><tbody>';
    df.preview.forEach(row => {
        html += '<tr>';
        df.columns.forEach(c => html += `<td>${row[c] !== null ? row[c] : '<span style="color:#999">NULL</span>'}</td>`);
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    preview.innerHTML = html;
    document.getElementById('upload-result').classList.remove('hidden');
}

// ==================== EDA 加载 ====================
async function loadEDA() {
    const loading = document.getElementById('eda-loading');
    const content = document.getElementById('eda-content');
    loading.classList.remove('hidden');
    content.classList.add('hidden');

    try {
        const [infoRes, edaRes] = await Promise.all([
            fetch('/api/data/info'),
            fetch('/api/data/eda')
        ]);
        const info = await infoRes.json();
        const eda = await edaRes.json();

        if (info.success) renderBasicInfo(info.info);
        if (eda.success) renderEDA(eda.eda);

        // 自动加载列质量分析
        loadColumnQuality();
        // 自动加载数据质量报告
        loadDataQualityReport();

        loading.classList.add('hidden');
        content.classList.remove('hidden');
    } catch (e) {
        showToast('EDA加载失败', 'error');
    }
}

function renderBasicInfo(info) {
    // 缓存列名和类型信息，供高级分析使用
    analyticsColumns = info.columns.map(c => c.column);
    if (!uploadedData) uploadedData = {};
    uploadedData.column_types = {};
    info.columns.forEach(c => {
        if (c.inferred_type === 'numeric') {
            uploadedData.column_types[c.column] = 'numeric';
        } else if (c.inferred_type === 'datetime') {
            uploadedData.column_types[c.column] = 'datetime';
        } else {
            uploadedData.column_types[c.column] = 'categorical';
        }
    });

    document.getElementById('basic-info').innerHTML = `
        <div class="config-display">
            <div class="config-item"><div class="label">行数</div><div class="value">${info.shape[0].toLocaleString()}</div></div>
            <div class="config-item"><div class="label">列数</div><div class="value">${info.shape[1]}</div></div>
            <div class="config-item"><div class="label">内存占用</div><div class="value">${info.memory_mb} MB</div></div>
        </div>
    `;

    // 类型分布
    const typeCounts = {};
    info.columns.forEach(c => { typeCounts[c.inferred_type] = (typeCounts[c.inferred_type] || 0) + 1; });
    let typeHtml = '';
    for (const [t, n] of Object.entries(typeCounts)) {
        typeHtml += `<div class="config-item"><div class="label">${t}</div><div class="value">${n} 列</div></div>`;
    }
    document.getElementById('type-distribution').innerHTML = `<div class="config-display">${typeHtml}</div>`;

    // 字段详情表
    const thead = document.querySelector('#column-table thead');
    const tbody = document.querySelector('#column-table tbody');
    thead.innerHTML = '<tr><th>列名</th><th>数据类型</th><th>推断类型</th><th>唯一值</th><th>缺失数</th><th>缺失率</th></tr>';
    tbody.innerHTML = info.columns.map(c => `
        <tr>
            <td><strong>${c.column}</strong></td>
            <td>${c.dtype}</td>
            <td><span class="badge badge-green">${c.inferred_type}</span></td>
            <td>${c.n_unique.toLocaleString()}</td>
            <td>${c.missing.toLocaleString()}</td>
            <td>${(c.missing_rate * 100).toFixed(2)}%</td>
        </tr>
    `).join('');
}

async function loadDataQualityReport() {
    const card = document.getElementById('data-quality-card');
    const container = document.getElementById('data-quality-content');
    if (!card || !container) return;
    const targetCol = document.getElementById('cfg-target') ? document.getElementById('cfg-target').value : null;
    const taskType = document.getElementById('cfg-task') ? document.getElementById('cfg-task').value : null;
    try {
        const res = await fetch('/api/data/quality', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_col: targetCol, task_type: taskType })
        });
        const data = await res.json();
        if (data.success && data.report) {
            renderDataQualityReport(data.report);
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    } catch (e) {
        card.style.display = 'none';
    }
}

async function autofixData() {
    const targetCol = document.getElementById('cfg-target') ? document.getElementById('cfg-target').value : null;
    const taskType = document.getElementById('cfg-task') ? document.getElementById('cfg-task').value : null;
    const fixOutliers = document.getElementById('autofix-outliers') ? document.getElementById('autofix-outliers').checked : false;
    try {
        const res = await fetch('/api/data/autofix', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_col: targetCol, task_type: taskType, fix_outliers: fixOutliers })
        });
        const data = await res.json();
        if (data.success) {
            showToast('Auto-fix applied: ' + data.fixes.join('; '), 'success');
            loadDataQualityReport();
            loadEDA();
        } else {
            showToast(data.error || 'Auto-fix failed', 'error');
        }
    } catch (e) {
        showToast('Auto-fix error: ' + e.message, 'error');
    }
}

function renderDataQualityReport(report) {
    const container = document.getElementById('data-quality-content');
    if (!container) return;
    function healthBar(label, pct, color) {
        return '<div style="margin:4px 0;"><div style="display:flex;justify-content:space-between;font-size:11px;"><span>' + label + '</span><span>' + pct + '%</span></div><div style="height:6px;background:#eee;border-radius:3px;overflow:hidden;"><div style="width:' + pct + '%;height:100%;background:' + color + ';"></div></div></div>';
    }
    const missingPct = report.n_rows > 0 ? Math.min(100, Math.round(report.missing_values.total_missing_cells / (report.n_rows * report.n_columns) * 100)) : 0;
    const dupPct = report.n_rows > 0 ? Math.min(100, Math.round(report.duplicates.duplicate_rows / report.n_rows * 100)) : 0;
    const outPct = report.n_rows > 0 ? Math.min(100, Math.round(report.outliers.total_outliers / (report.n_rows * report.n_columns) * 100)) : 0;
    let html = '<div class="config-display">';
    html += '<div class="config-item"><div class="label">Rows</div><div class="value">' + report.n_rows.toLocaleString() + '</div></div>';
    html += '<div class="config-item"><div class="label">Cols</div><div class="value">' + report.n_columns + '</div></div>';
    html += '<div class="config-item"><div class="label">Memory</div><div class="value">' + report.memory_mb + ' MB</div></div>';
    html += '</div>';
    html += '<div style="margin-top:10px;">';
    html += healthBar('Missing Health', 100 - missingPct, missingPct > 20 ? '#e74c3c' : (missingPct > 5 ? '#f39c12' : '#27ae60'));
    html += healthBar('Duplicate Health', 100 - dupPct, dupPct > 10 ? '#e74c3c' : (dupPct > 2 ? '#f39c12' : '#27ae60'));
    html += healthBar('Outlier Health', 100 - outPct, outPct > 10 ? '#e74c3c' : (outPct > 3 ? '#f39c12' : '#27ae60'));
    html += '</div>';
    if (report.missing_values.columns_with_missing > 0) {
        html += '<h4 style="margin:12px 0 6px;font-size:13px;">Missing Values</h4>';
        html += '<table class="data-table" style="font-size:12px;"><thead><tr><th>Column</th><th>Count</th><th>%</th><th>Suggestion</th></tr></thead><tbody>';
        report.missing_values.details.forEach(d => {
            html += '<tr><td>' + d.column + '</td><td>' + d.missing_count + '</td><td>' + d.missing_percent + '%</td><td>' + d.suggestion + '</td></tr>';
        });
        html += '</tbody></table>';
    }
    if (report.outliers.columns_with_outliers > 0) {
        html += '<h4 style="margin:12px 0 6px;font-size:13px;">Outliers (Top)</h4>';
        html += '<table class="data-table" style="font-size:12px;"><thead><tr><th>Column</th><th>Count</th><th>%</th></tr></thead><tbody>';
        report.outliers.details.forEach(d => {
            html += '<tr><td>' + d.column + '</td><td>' + d.outlier_count + '</td><td>' + d.outlier_percent + '%</td></tr>';
        });
        html += '</tbody></table>';
    }
    if (report.constant_columns && report.constant_columns.count > 0) {
        html += '<h4 style="margin:12px 0 6px;font-size:13px;color:#e74c3c;">Constant Columns (' + report.constant_columns.count + ')</h4>';
        html += '<p style="font-size:12px;">' + report.constant_columns.columns.map(c => c.column).join(', ') + ' — consider dropping</p>';
    }
    if (report.high_cardinality && report.high_cardinality.count > 0) {
        html += '<h4 style="margin:12px 0 6px;font-size:13px;color:#e67e22;">High Cardinality (' + report.high_cardinality.count + ')</h4>';
        html += '<p style="font-size:12px;">' + report.high_cardinality.columns.map(c => c.column + ' (' + c.unique_ratio + ')').join(', ') + ' — consider encoding or dropping</p>';
    }
    if (report.target_leakage && report.target_leakage.count > 0) {
        html += '<h4 style="margin:12px 0 6px;font-size:13px;color:#c0392b;">Target Leakage Warning (' + report.target_leakage.count + ')</h4>';
        html += '<p style="font-size:12px;">' + report.target_leakage.columns.map(c => c.column + ': ' + c.reason).join('; ') + '</p>';
    }
    if (report.target) {
        html += '<h4 style="margin:12px 0 6px;font-size:13px;">Target: ' + report.target.type + '</h4>';
        if (report.target.type === 'classification') {
            html += '<p style="font-size:12px;">Classes: ' + report.target.n_classes + ' | Imbalance ratio: ' + report.target.imbalance_ratio + ' | ' + report.target.suggestion + '</p>';
        } else {
            html += '<p style="font-size:12px;">Mean: ' + report.target.mean + ' | Std: ' + report.target.std + ' | Skew: ' + report.target.skewness + ' | ' + report.target.suggestion + '</p>';
        }
    }
    container.innerHTML = html;
}

function renderEDA(eda) {
    // 相关性热力图
    if (eda.correlation && eda.correlation.columns) {
        const chartDom = document.getElementById('correlation-chart');
        const myChart = echarts.init(chartDom);
        const cols = eda.correlation.columns;
        const vals = eda.correlation.values;
        const data = [];
        for (let i = 0; i < cols.length; i++) {
            for (let j = 0; j < cols.length; j++) {
                data.push([j, i, parseFloat(vals[i][j].toFixed(2))]);
            }
        }
        myChart.setOption({
            tooltip: { position: 'top' },
            grid: { height: '70%', top: '10%' },
            xAxis: { type: 'category', data: cols, splitArea: { show: true }, axisLabel: { rotate: 45 } },
            yAxis: { type: 'category', data: cols, splitArea: { show: true } },
            visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: '0%', inRange: { color: ['#d73027', '#f7f7f7', '#1a9850'] } },
            series: [{ name: '相关性', type: 'heatmap', data: data, label: { show: true }, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } } }]
        });
    }
}

// ==================== 高级统计分析 ====================

let analyticsColumns = [];  // 缓存当前数据的列名

function onAnalyticsTypeChange() {
    const type = document.getElementById('analytics-type').value;
    const paramsEl = document.getElementById('analytics-params');
    if (!type) {
        paramsEl.innerHTML = '';
        return;
    }
    
    const numCols = analyticsColumns.filter(c => uploadedData && uploadedData.column_types && uploadedData.column_types[c] === 'numeric');
    const catCols = analyticsColumns.filter(c => uploadedData && uploadedData.column_types && uploadedData.column_types[c] === 'categorical');
    // 卡方检验排除高基数列（唯一值太多没有意义，且会生成巨大列联表）
    const chi2Cols = catCols.filter(c => {
        const dist = uploadedData.distributions && uploadedData.distributions[c];
        if (!dist) return true;
        const unique = dist.unique_count || 0;
        return unique <= 30 && unique >= 2;
    });
    
    let html = '';
    switch (type) {
        case 'descriptive':
            html = `<label>选择列（留空则全部数值列）</label><select id="analytics-columns" multiple style="height:80px;">${numCols.map(c => `<option value="${c}">${c}</option>`).join('')}</select>`;
            break;
        case 'pca':
            html = `<label>主成分数</label><input type="number" id="analytics-ncomp" value="3" min="2" max="10" style="width:80px;">`;
            break;
        case 'factor':
            html = `<label>因子数（留空自动）</label><input type="number" id="analytics-nfactors" value="" min="1" max="10" style="width:80px;" placeholder="自动"><label style="margin-left:10px;">旋转</label><select id="analytics-rotation"><option value="varimax">Varimax</option><option value="none">无</option></select>`;
            break;
        case 'correlation':
            html = `<label>方法</label><select id="analytics-method"><option value="pearson">Pearson</option><option value="spearman">Spearman</option><option value="kendall">Kendall</option></select><label style="margin-left:10px;">阈值</label><input type="number" id="analytics-threshold" value="0.5" min="0" max="1" step="0.1" style="width:60px;">`;
            break;
        case 'anova':
            html = `<label>因子（分类）</label><select id="analytics-factor">${catCols.map(c => `<option value="${c}">${c}</option>`).join('')}</select><label style="margin-left:10px;">目标（数值）</label><select id="analytics-target">${numCols.map(c => `<option value="${c}">${c}</option>`).join('')}</select>`;
            break;
        case 'chi2':
            html = `<label>列1</label><select id="analytics-col1">${chi2Cols.map(c => `<option value="${c}">${c}</option>`).join('')}</select><label style="margin-left:10px;">列2</label><select id="analytics-col2">${chi2Cols.map(c => `<option value="${c}">${c}</option>`).join('')}</select>`;
            break;
        case 'outliers':
            html = `<label>列</label><select id="analytics-column">${numCols.map(c => `<option value="${c}">${c}</option>`).join('')}</select><label style="margin-left:10px;">方法</label><select id="analytics-omethod"><option value="iqr">IQR</option><option value="zscore">Z-Score</option></select>`;
            break;
    }
    paramsEl.innerHTML = html;
}

async function runAdvancedAnalytics() {
    const type = document.getElementById('analytics-type').value;
    if (!type) {
        showToast('请选择分析类型', 'error');
        return;
    }
    
    const resultEl = document.getElementById('analytics-result');
    resultEl.innerHTML = '<div class="loading">分析中...</div>';
    
    let payload = {};
    switch (type) {
        case 'descriptive':
            const cols = document.getElementById('analytics-columns');
            payload = { columns: cols ? Array.from(cols.selectedOptions).map(o => o.value) : null };
            break;
        case 'pca':
            payload = { n_components: parseInt(document.getElementById('analytics-ncomp').value) || 3 };
            break;
        case 'factor':
            const nf = document.getElementById('analytics-nfactors').value;
            payload = { n_factors: nf ? parseInt(nf) : null, rotation: document.getElementById('analytics-rotation').value };
            break;
        case 'correlation':
            payload = { method: document.getElementById('analytics-method').value, threshold: parseFloat(document.getElementById('analytics-threshold').value) };
            break;
        case 'anova':
            payload = { factor: document.getElementById('analytics-factor').value, target: document.getElementById('analytics-target').value };
            break;
        case 'chi2':
            payload = { col1: document.getElementById('analytics-col1').value, col2: document.getElementById('analytics-col2').value };
            break;
        case 'outliers':
            payload = { column: document.getElementById('analytics-column').value, method: document.getElementById('analytics-omethod').value };
            break;
    }
    
    try {
        const res = await fetch(`/api/analytics/${type}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            renderAnalyticsResult(type, data);
        } else {
            resultEl.innerHTML = `<div style="color:#c00;">❌ ${escapeHtml(data.error || '分析失败')}</div>`;
        }
    } catch (e) {
        resultEl.innerHTML = `<div style="color:#c00;">❌ 请求失败: ${escapeHtml(e.message)}</div>`;
    }
}

function renderAnalyticsResult(type, data) {
    const el = document.getElementById('analytics-result');
    switch (type) {
        case 'descriptive':
            renderDescriptiveResult(el, data.stats);
            break;
        case 'pca':
            renderPCAResult(el, data.pca);
            break;
        case 'factor':
            renderFactorResult(el, data.factor);
            break;
        case 'correlation':
            renderCorrelationResult(el, data.correlation);
            break;
        case 'anova':
            renderAnovaResult(el, data.anova);
            break;
        case 'chi2':
            renderChi2Result(el, data.chi2);
            break;
        case 'outliers':
            renderOutlierResult(el, data.outliers);
            break;
    }
}

function renderDescriptiveResult(el, stats) {
    if (!stats || stats.length === 0) {
        el.innerHTML = '<div class="hint">无数值列可分析</div>';
        return;
    }
    let html = '<div class="table-wrapper"><table class="data-table"><thead><tr><th>列名</th><th>样本数</th><th>缺失</th><th>均值</th><th>标准差</th><th>最小值</th><th>中位数</th><th>最大值</th><th>Q1</th><th>Q3</th><th>偏度</th><th>峰度</th><th>95%CI</th></tr></thead><tbody>';
    stats.forEach(s => {
        const ci = (s.ci_lower !== null && s.ci_upper !== null) ? `[${s.ci_lower}, ${s.ci_upper}]` : '-';
        html += `<tr><td><strong>${escapeHtml(s.column)}</strong></td><td>${s.count}</td><td>${s.missing}(${Math.round(s.missing_rate*100)}%)</td><td>${s.mean ?? '-'}</td><td>${s.std ?? '-'}</td><td>${s.min ?? '-'}</td><td>${s.median ?? '-'}</td><td>${s.max ?? '-'}</td><td>${s.q1 ?? '-'}</td><td>${s.q3 ?? '-'}</td><td>${s.skewness ?? '-'}</td><td>${s.kurtosis ?? '-'}</td><td style="font-size:11px;">${ci}</td></tr>`;
    });
    html += '</tbody></table></div>';
    el.innerHTML = html;
}

function renderPCAResult(el, pca) {
    let html = `<div style="margin-bottom:12px;"><strong>原始特征数:</strong> ${pca.original_features} · <strong>提取主成分:</strong> ${pca.n_components}</div>`;
    
    // 方差解释表
    html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>主成分</th><th>方差解释率</th><th>累计方差</th><th>主要贡献特征</th></tr></thead><tbody>';
    pca.components.forEach((comp, i) => {
        const features = Object.entries(comp.features).map(([k, v]) => `${escapeHtml(k)}(${v > 0 ? '+' : ''}${v})`).join(', ');
        html += `<tr><td>${comp.pc}</td><td>${(comp.variance_ratio * 100).toFixed(2)}%</td><td>${(pca.cumulative_variance[i] * 100).toFixed(2)}%</td><td style="font-size:12px;">${features}</td></tr>`;
    });
    html += '</tbody></table></div>';
    
    // 方差解释柱状图
    html += '<div id="pca-chart" style="width:100%;height:280px;margin-top:16px;"></div>';
    el.innerHTML = html;
    
    setTimeout(() => {
        const chart = echarts.init(document.getElementById('pca-chart'));
        chart.setOption({
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            legend: { data: ['方差解释率', '累计方差'] },
            xAxis: { type: 'category', data: pca.components.map(c => c.pc) },
            yAxis: { type: 'value', max: 1, axisLabel: { formatter: v => (v * 100).toFixed(0) + '%' } },
            series: [
                { name: '方差解释率', type: 'bar', data: pca.explained_variance_ratio },
                { name: '累计方差', type: 'line', data: pca.cumulative_variance }
            ]
        });
    }, 0);
}

function renderFactorResult(el, factor) {
    let html = '';
    
    // 适用性检验
    const kmoColor = factor.kmo_acceptable ? '#2e7d32' : '#c00';
    const bartlettColor = factor.bartlett_significant ? '#2e7d32' : '#c00';
    html += `<div style="margin-bottom:12px;display:flex;gap:20px;flex-wrap:wrap;">`;
    html += `<div style="background:#f8f9fa;padding:10px 14px;border-radius:6px;"><strong>KMO检验:</strong> ${factor.kmo !== null ? factor.kmo : 'N/A'} <span style="color:${kmoColor};">${factor.kmo_acceptable ? '✓ 适合' : '✗ 不适合'}</span></div>`;
    html += `<div style="background:#f8f9fa;padding:10px 14px;border-radius:6px;"><strong>Bartlett检验:</strong> χ²=${factor.bartlett_chi2 !== null ? factor.bartlett_chi2 : 'N/A'}, P=${factor.bartlett_pvalue !== null ? factor.bartlett_pvalue : 'N/A'} <span style="color:${bartlettColor};">${factor.bartlett_significant ? '✓ 显著' : '✗ 不显著'}</span></div>`;
    html += `<div style="background:#f8f9fa;padding:10px 14px;border-radius:6px;"><strong>提取因子:</strong> ${factor.n_factors} · <strong>旋转:</strong> ${factor.rotation || '无'}</div>`;
    html += `</div>`;
    
    // 因子载荷矩阵
    html += '<h4 style="margin:16px 0 8px;">因子载荷矩阵</h4>';
    html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>特征</th>';
    for (let i = 1; i <= factor.n_factors; i++) html += `<th>F${i}</th>`;
    html += '<th>主导因子</th><th>共同度</th></tr></thead><tbody>';
    factor.loadings.forEach(row => {
        html += `<tr><td><strong>${escapeHtml(row.feature)}</strong></td>`;
        for (let i = 1; i <= factor.n_factors; i++) {
            const val = row[`F${i}`];
            const color = Math.abs(val) >= 0.5 ? '#2e7d32' : (Math.abs(val) >= 0.3 ? '#f60' : '#999');
            html += `<td style="color:${color};font-weight:${Math.abs(val) >= 0.5 ? 600 : 400};">${val}</td>`;
        }
        html += `<td>${row.dominant_factor}</td><td>${factor.communalities[row.feature]}</td></tr>`;
    });
    html += '</tbody></table></div>';
    
    // 方差解释
    html += '<h4 style="margin:16px 0 8px;">方差解释</h4>';
    html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>因子</th><th>平方和</th><th>比例</th><th>累计</th></tr></thead><tbody>';
    factor.variance_explained.forEach(v => {
        html += `<tr><td>${v.factor}</td><td>${v.sum_of_squares}</td><td>${(v.proportion * 100).toFixed(2)}%</td><td>${(v.cumulative * 100).toFixed(2)}%</td></tr>`;
    });
    html += '</tbody></table></div>';
    
    // 因子载荷热力图
    html += '<div id="factor-chart" style="width:100%;height:300px;margin-top:16px;"></div>';
    el.innerHTML = html;
    
    setTimeout(() => {
        const chartDom = document.getElementById('factor-chart');
        if (!chartDom) return;
        const chart = echarts.init(chartDom);
        const features = factor.loadings.map(l => l.feature);
        const factors = factor.loadings[0] ? Object.keys(factor.loadings[0]).filter(k => /^F\d+$/.test(k)) : [];
        const heatData = [];
        features.forEach((feat, i) => {
            factors.forEach((f, j) => {
                heatData.push([j, i, factor.loadings[i][f]]);
            });
        });
        chart.setOption({
            tooltip: { position: 'top', formatter: p => `${features[p.data[1]]} · ${factors[p.data[0]]}: ${p.data[2]}` },
            grid: { height: '70%', top: '10%', left: '20%', right: '10%' },
            xAxis: { type: 'category', data: factors, splitArea: { show: true } },
            yAxis: { type: 'category', data: features, splitArea: { show: true } },
            visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: '0%', inRange: { color: ['#d73027', '#f7f7f7', '#1a9850'] } },
            series: [{ name: '载荷', type: 'heatmap', data: heatData, label: { show: true, fontSize: 11 }, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } } }]
        });
    }, 0);
}

// ==================== 数据筛选与自动分析 ====================

let columnQualityData = [];  // 缓存列质量数据
let rowFilterConditions = []; // 行筛选条件

async function loadColumnQuality() {
    const listEl = document.getElementById('column-quality-list');
    listEl.innerHTML = '<div class="loading">正在分析列质量...</div>';
    try {
        const res = await fetch('/api/data/column-quality');
        const data = await res.json();
        if (data.success) {
            columnQualityData = data.quality;
            renderColumnQuality(data.quality);
            renderRowFilterSelectors();
        } else {
            listEl.innerHTML = `<div style="color:#c00;">❌ ${escapeHtml(data.error)}</div>`;
        }
    } catch (e) {
        listEl.innerHTML = `<div style="color:#c00;">❌ 加载失败: ${escapeHtml(e.message)}</div>`;
    }
}

function renderColumnQuality(quality) {
    const listEl = document.getElementById('column-quality-list');
    if (!quality || quality.length === 0) {
        listEl.innerHTML = '<div class="hint">无数值列可分析</div>';
        return;
    }
    
    let html = '<div class="table-wrapper"><table class="data-table" style="font-size:12px;"><thead><tr><th><input type="checkbox" id="cq-select-all" checked onchange="toggleAllColumns(this)"></th><th>列名</th><th>类型</th><th>缺失率</th><th>唯一值</th><th>方差</th><th>建议</th><th>原因</th></tr></thead><tbody>';
    quality.forEach(q => {
        const recColor = q.recommendation === 'drop' ? '#c00' : (q.recommendation === 'review' ? '#f60' : '#2e7d32');
        const recText = q.recommendation === 'drop' ? '删除' : (q.recommendation === 'review' ? '审查' : '保留');
        const rowClass = q.recommendation === 'drop' ? 'style="background:#ffebee;"' : '';
        html += `<tr ${rowClass}><td><input type="checkbox" class="cq-col-check" value="${escapeHtml(q.column)}" ${q.recommendation !== 'drop' ? 'checked' : ''}></td><td><strong>${escapeHtml(q.column)}</strong></td><td>${q.dtype}</td><td>${(q.missing_rate * 100).toFixed(1)}%</td><td>${q.unique}</td><td>${q.variance !== null ? q.variance.toExponential(2) : '-'}</td><td style="color:${recColor};font-weight:600;">${recText}</td><td style="font-size:11px;color:#999;">${q.reasons.join(', ') || '-'}</td></tr>`;
    });
    html += '</tbody></table></div>';
    listEl.innerHTML = html;
}

function toggleAllColumns(checkbox) {
    document.querySelectorAll('.cq-col-check').forEach(cb => {
        cb.checked = checkbox.checked;
    });
}

function renderRowFilterSelectors() {
    // 为行筛选提供列选项
    const cols = columnQualityData.map(q => q.column);
    window._filterColumns = cols;
}

function addRowFilter() {
    const cols = window._filterColumns || analyticsColumns || [];
    if (cols.length === 0) {
        showToast('请先上传数据', 'error');
        return;
    }
    const id = Date.now() + Math.random();
    const filterHtml = `
        <div class="row-filter-item" data-id="${id}" style="display:flex;gap:8px;align-items:center;margin-bottom:8px;padding:8px;background:#f8f9fa;border-radius:6px;">
            <select class="rf-col" style="min-width:120px;">${cols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')}</select>
            <select class="rf-op" style="min-width:100px;">
                <option value="eq">等于</option>
                <option value="ne">不等于</option>
                <option value="gt">大于</option>
                <option value="gte">大于等于</option>
                <option value="lt">小于</option>
                <option value="lte">小于等于</option>
                <option value="contains">包含</option>
                <option value="startswith">开头是</option>
                <option value="endswith">结尾是</option>
                <option value="isnull">为空</option>
                <option value="notnull">非空</option>
            </select>
            <input type="text" class="rf-val" placeholder="值" style="width:120px;">
            <button class="btn btn-sm" style="padding:4px 8px;font-size:12px;color:#c00;" onclick="this.closest('.row-filter-item').remove()">✕</button>
        </div>
    `;
    const container = document.getElementById('row-filters-list');
    const wrapper = document.createElement('div');
    wrapper.innerHTML = filterHtml;
    container.appendChild(wrapper.firstElementChild);
}

async function applyDataFilter() {
    // 收集选中的列
    const selectedCols = Array.from(document.querySelectorAll('.cq-col-check:checked')).map(cb => cb.value);
    if (selectedCols.length === 0) {
        showToast('请至少选择一列', 'error');
        return;
    }
    
    // 收集行筛选条件
    const rowFilters = [];
    document.querySelectorAll('.row-filter-item').forEach(item => {
        const col = item.querySelector('.rf-col').value;
        const op = item.querySelector('.rf-op').value;
        const val = item.querySelector('.rf-val').value;
        if (op === 'isnull' || op === 'notnull') {
            rowFilters.push({ column: col, operator: op, value: null });
        } else if (val !== '') {
            rowFilters.push({ column: col, operator: op, value: val });
        }
    });
    
    try {
        const res = await fetch('/api/data/filter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ columns: selectedCols, row_filters: rowFilters })
        });
        const data = await res.json();
        if (data.success) {
            uploadedData = data.data;
            document.getElementById('filter-status').style.display = 'block';
            document.getElementById('filter-status').innerHTML = `✅ 已筛选: <strong>${data.shape[0]}</strong> 行 × <strong>${data.shape[1]}</strong> 列 (原 ${data.row_count_before} 行)`;
            document.getElementById('data-status').textContent = `已加载: ${data.shape[0]}行×${data.shape[1]}列 (已筛选)`;
            populateTargetOptions(data.data.columns, data.target_hint);
            showToast(`筛选成功: ${data.shape[0]} 行 × ${data.shape[1]} 列`);
            // 清除旧的EDA结果，重新加载
            document.getElementById('eda-content').classList.add('hidden');
            loadEDA();
        } else {
            showToast(data.error || '筛选失败', 'error');
        }
    } catch (e) {
        showToast('筛选出错: ' + e.message, 'error');
    }
}

async function resetDataFilter() {
    try {
        const res = await fetch('/api/data/reset-filter', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            uploadedData = data.data;
            document.getElementById('filter-status').style.display = 'none';
            document.getElementById('data-status').textContent = `已加载: ${data.shape[0]}行×${data.shape[1]}列`;
            populateTargetOptions(data.data.columns, data.target_hint);
            showToast('已恢复原始数据');
            // 重新加载列质量分析
            loadColumnQuality();
            // 清除行筛选条件
            document.getElementById('row-filters-list').innerHTML = '';
            // 重新加载EDA
            document.getElementById('eda-content').classList.add('hidden');
            loadEDA();
        } else {
            showToast(data.error || '重置失败', 'error');
        }
    } catch (e) {
        showToast('重置出错: ' + e.message, 'error');
    }
}

function renderCorrelationResult(el, corr) {
    let html = `<div style="margin-bottom:10px;"><strong>方法:</strong> ${corr.method} · <strong>显著相关对:</strong> ${corr.pairs.length}</div>`;
    if (corr.pairs.length > 0) {
        html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>列1</th><th>列2</th><th>相关系数</th><th>强度</th><th>P值</th></tr></thead><tbody>';
        corr.pairs.forEach(p => {
            const color = p.abs_correlation >= 0.8 ? '#c00' : (p.abs_correlation >= 0.5 ? '#f60' : '#999');
            html += `<tr><td>${escapeHtml(p.col1)}</td><td>${escapeHtml(p.col2)}</td><td style="color:${color};font-weight:600;">${p.correlation}</td><td>${p.strength}</td><td>${p.p_value ?? '-'}</td></tr>`;
        });
        html += '</tbody></table></div>';
    } else {
        html += '<div class="hint">未找到显著相关对（阈值以上）</div>';
    }
    el.innerHTML = html;
}

function renderAnovaResult(el, anova) {
    const sigColor = anova.significant ? '#2e7d32' : '#999';
    let html = `<div style="margin-bottom:10px;"><strong>因子:</strong> ${escapeHtml(anova.factor)} · <strong>目标:</strong> ${escapeHtml(anova.target)} · <strong>F值:</strong> ${anova.f_statistic} · <strong>P值:</strong> <span style="color:${sigColor};font-weight:600;">${anova.p_value}</span> · <strong>显著:</strong> <span style="color:${sigColor}">${anova.significant ? '是 ✓' : '否'}</span></div>`;
    html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>分组</th><th>样本数</th><th>均值</th><th>标准差</th></tr></thead><tbody>';
    anova.group_stats.forEach(g => {
        html += `<tr><td>${escapeHtml(g.group)}</td><td>${g.count}</td><td>${g.mean}</td><td>${g.std}</td></tr>`;
    });
    html += '</tbody></table></div>';
    el.innerHTML = html;
}

function renderChi2Result(el, chi2) {
    const sigColor = chi2.significant ? '#2e7d32' : '#999';
    let html = `<div style="margin-bottom:10px;"><strong>列1:</strong> ${escapeHtml(chi2.col1)} · <strong>列2:</strong> ${escapeHtml(chi2.col2)} · <strong>χ²:</strong> ${chi2.chi2} · <strong>自由度:</strong> ${chi2.dof} · <strong>P值:</strong> <span style="color:${sigColor};font-weight:600;">${chi2.p_value}</span> · <strong>Cramér's V:</strong> ${chi2.cramers_v} · <strong>效应量:</strong> ${escapeHtml(chi2.effect_size || '-')} · <strong>显著:</strong> <span style="color:${sigColor}">${chi2.significant ? '是 ✓' : '否'}</span></div>`;

    // 列联表
    if (chi2.contingency_table) {
        if (chi2.table_warning) {
            html += `<div class="hint">${escapeHtml(chi2.table_warning)}</div>`;
        }
        const rows = Object.keys(chi2.contingency_table);
        if (rows.length > 0) {
            const cols = Object.keys(chi2.contingency_table[rows[0]]);
            html += '<div class="table-wrapper" style="max-height:400px;overflow:auto;"><table class="data-table"><thead><tr><th></th>';
            cols.forEach(c => html += `<th>${escapeHtml(String(c))}</th>`);
            html += '</tr></thead><tbody>';
            rows.forEach(r => {
                html += `<tr><th>${escapeHtml(String(r))}</th>`;
                cols.forEach(c => {
                    const val = chi2.contingency_table[r][c];
                    html += `<td>${val !== undefined ? val : ''}</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody></table></div>';
        }
    }
    el.innerHTML = html;
}

function renderOutlierResult(el, out) {
    let html = `<div style="margin-bottom:10px;"><strong>列:</strong> ${escapeHtml(out.column)} · <strong>方法:</strong> ${out.method} · <strong>异常值:</strong> ${out.outlier_count} / ${out.total} (${(out.outlier_rate * 100).toFixed(2)}%)</div>`;
    if (out.method === 'iqr') {
        html += `<div class="hint">正常范围: [${out.bounds.lower.toFixed(2)}, ${out.bounds.upper.toFixed(2)}] (Q1=${out.bounds.q1.toFixed(2)}, Q3=${out.bounds.q3.toFixed(2)})</div>`;
    } else {
        html += `<div class="hint">阈值: |Z| > 3 (均值=${out.bounds.mean.toFixed(2)}, 标准差=${out.bounds.std.toFixed(2)})</div>`;
    }
    if (out.outliers.length > 0) {
        html += '<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;">';
        out.outliers.forEach(o => {
            html += `<span style="background:#ffebee;padding:2px 8px;border-radius:4px;font-size:12px;color:#c00;">${o.value}</span>`;
        });
        html += '</div>';
    }
    el.innerHTML = html;
}

// ==================== 建模配置 ====================
function populateTargetOptions(columns, hint) {
    const sel = document.getElementById('cfg-target');
    // 支持多选：hint 可以是字符串或数组
    const hints = Array.isArray(hint) ? hint : (hint ? [hint] : []);
    sel.innerHTML = '<option value="">请选择目标列（聚类可不选）</option>' + columns.map(c =>
        `<option value="${c}" ${hints.includes(c) ? 'selected' : ''}>${c}</option>`
    ).join('');
}

function initModeHint() {
    const modeSel = document.getElementById('cfg-decision-mode');
    const hints = {
        balanced: '综合考虑精度、速度、稳定性，适合大多数场景',
        accuracy_first: '选择CV分数最高的模型，适合比赛打榜',
        speed_first: '选择训练快、推断快的模型，适合实时应用',
        stability_first: '选择CV方差最小的模型，适合生产环境',
        simplicity_first: '选择复杂度低的模型，防止过拟合',
    };
    modeSel.addEventListener('change', () => {
        document.getElementById('mode-hint').textContent = hints[modeSel.value] || '';
    });
}

async function loadAutoConfig() {
    // 基于当前数据生成自动推荐配置展示
    const target = document.getElementById('cfg-target').value;
    const mode = document.getElementById('cfg-decision-mode').value;
    const modeName = document.querySelector(`#cfg-decision-mode option[value="${mode}"]`).text;

    const display = document.getElementById('auto-config-display');
    display.innerHTML = `
        <div class="config-item"><div class="label">目标列</div><div class="value">${target || '自动推断（聚类）'}</div></div>
        <div class="config-item"><div class="label">编码策略</div><div class="value">自动（OneHot/Label/Target）</div></div>
        <div class="config-item"><div class="label">特征选择</div><div class="value">互信息 (MI)</div></div>
        <div class="config-item"><div class="label">决策模式</div><div class="value">${modeName}</div></div>
        <div class="config-item"><div class="label">融合策略</div><div class="value">加权平均</div></div>
        <div class="config-item"><div class="label">CV折数</div><div class="value">5折</div></div>
        <div class="config-item"><div class="label">降采样</div><div class="value">最大50,000样本</div></div>
    `;
}

function acceptAutoConfig() {
    showManualConfig();
    showToast('已接受推荐配置，您可以进一步微调或直接开始训练');
}

function showManualConfig() {
    document.getElementById('auto-recommend-panel').classList.add('hidden');
    document.getElementById('manual-config-panel').classList.remove('hidden');
}

function hideManualConfig() {
    document.getElementById('manual-config-panel').classList.add('hidden');
    document.getElementById('auto-recommend-panel').classList.remove('hidden');
}

// ==================== 训练 ====================
async function startTraining() {
    const config = collectConfig();
    goStep(5);

    document.getElementById('train-status-panel').classList.remove('hidden');
    document.getElementById('train-result').classList.add('hidden');
    document.getElementById('train-spinner').classList.remove('hidden');
    const logPanel = document.getElementById('train-log-panel');
    if (logPanel) logPanel.innerHTML = '';
    const liveBoard = document.getElementById('live-leaderboard-content');
    if (liveBoard) liveBoard.innerHTML = '<p style="color:var(--text-light);font-size:13px;text-align:center;padding:20px;">Waiting for models...</p>';
    document.getElementById('train-progress-fill').style.width = '0%';
    document.getElementById('train-progress-detail').textContent = '';
    document.getElementById('train-status-text').textContent = '正在训练模型，请稍候...';
    trainEventSinceId = -1;

    try {
        const res = await fetch('/api/model/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (data.success) {
            pollTrainingStatus();
        } else {
            showToast(data.error || '训练启动失败', 'error');
            document.getElementById('train-spinner').classList.add('hidden');
        }
    } catch (e) {
        showToast('训练请求失败: ' + e.message, 'error');
        document.getElementById('train-spinner').classList.add('hidden');
    }
}

function collectConfig() {
    const targetSel = document.getElementById('cfg-target');
    const selectedTargets = Array.from(targetSel.selectedOptions).map(o => o.value).filter(v => v);
    return {
        target_col: selectedTargets.length > 0 ? (selectedTargets.length === 1 ? selectedTargets[0] : selectedTargets) : null,
        task_type: document.getElementById('cfg-task').value || null,
        auto_decision_mode: document.getElementById('cfg-decision-mode').value,
        encoding: document.getElementById('cfg-encoding').value,
        feature_selection: document.getElementById('cfg-fs').value,
        ensemble: document.getElementById('cfg-ensemble').value,
        feature_engineering: document.getElementById('cfg-feature-engineering').value === 'true',
        pseudo_labeling: document.getElementById('cfg-pseudo-labeling').value === 'true',
        fold_type: document.getElementById('cfg-fold-type').value,
        n_splits: parseInt(document.getElementById('cfg-cv').value),
        max_samples: parseInt(document.getElementById('cfg-max-samples').value),
        auto_sample: true,
        optimize_hyperparams: document.getElementById('cfg-hyperopt').checked,
        model_keys: document.getElementById('cfg-models').value
            ? document.getElementById('cfg-models').value.split(/[,，]/).map(s => s.trim()).filter(Boolean)
            : null,
        deep_learning: {
            enabled: document.getElementById('cfg-dl-mlp').checked || document.getElementById('cfg-dl-cnn').checked || document.getElementById('cfg-dl-lstm').checked,
            models: [
                ...(document.getElementById('cfg-dl-mlp').checked ? ['torch_mlp'] : []),
                ...(document.getElementById('cfg-dl-cnn').checked ? ['torch_cnn1d'] : []),
                ...(document.getElementById('cfg-dl-lstm').checked ? ['torch_lstm'] : []),
            ]
        },
        optimizer: document.getElementById('cfg-optimizer').value,
        dim_reduction: document.getElementById('cfg-dim-reduction').value,
    };
}

function pollTrainingStatus() {
    if (trainTimer) clearInterval(trainTimer);
    const progressFill = document.getElementById('train-progress-fill');
    const progressDetail = document.getElementById('train-progress-detail');
    const statusText = document.getElementById('train-status-text');
    trainTimer = setInterval(async () => {
        try {
            // 同时查询状态、事件和实时结果
            const [statusRes, eventsRes, liveRes] = await Promise.all([
                fetch('/api/model/status'),
                fetch(`/api/model/train-events?since_id=${trainEventSinceId}`),
                fetch('/api/model/live-results')
            ]);
            const statusData = await statusRes.json();
            const eventsData = await eventsRes.json();
            const liveData = await liveRes.json();

            if (!statusData.success) return;

            // 渲染实时事件日志
            if (eventsData.success && eventsData.events && eventsData.events.length > 0) {
                renderTrainEvents(eventsData.events);
                trainEventSinceId = eventsData.latest_id;
            }

            // 渲染实时排行榜
            if (liveData.success && liveData.results) {
                renderLiveLeaderboard(liveData.results);
            }

            if (statusData.status === 'done') {
                clearInterval(trainTimer);
                document.getElementById('train-spinner').classList.add('hidden');
                await loadTrainResult();
                document.getElementById('train-result').classList.remove('hidden');
                markStepCompleted(5);
                showToast('训练完成！');
            } else if (statusData.status === 'error') {
                clearInterval(trainTimer);
                document.getElementById('train-spinner').classList.add('hidden');
                const logPanel = document.getElementById('train-log-panel');
                if (logPanel) {
                    logPanel.innerHTML += `<div class="log-entry" style="color:#ff6b6b">[ERROR] ${statusData.error}</div>`;
                    logPanel.scrollTop = logPanel.scrollHeight;
                }
                showToast('训练失败: ' + statusData.error, 'error');
            } else if (statusData.status === 'running') {
                // 更新进度条
                if (statusData.progress) {
                    const p = statusData.progress;
                    const pct = p.percent || 0;
                    progressFill.style.width = pct + '%';
                    progressDetail.textContent = `${p.message || 'Training...'} (${p.current || 0}/${p.total || 0})`;
                    const stepMap = {
                        'preprocessing': 'Preprocessing data...',
                        'encoding': 'Encoding features...',
                        'feature_selection': 'Selecting features...',
                        'dim_reduction': 'Reducing dimensions...',
                        'hyperopt': 'Optimizing hyperparameters...',
                        'hyperopt_model': 'Optimizing hyperparameters...',
                        'training': 'Training models...',
                        'cv_fold': 'Cross-validating...',
                        'model_done': 'Training models...',
                        'ensemble': 'Blending ensemble...',
                        'done': 'Finalizing...'
                    };
                    statusText.textContent = stepMap[p.step] || 'Training...';
                }
            }
        } catch (e) {
            console.error(e);
        }
    }, 2000);
}

function renderTrainEvents(events) {
    const panel = document.getElementById('train-log-panel');
    if (!panel) return;
    const fragment = document.createDocumentFragment();
    for (const ev of events) {
        const div = document.createElement('div');
        div.className = 'log-entry';
        const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const stepClass = `log-step-${ev.step}`;
        div.innerHTML = `<span class="log-time">[${timeStr}]</span><span class="${stepClass}">${escapeHtml(ev.message)}</span>`;
        fragment.appendChild(div);
    }
    panel.appendChild(fragment);
    panel.scrollTop = panel.scrollHeight;
}

function renderLiveLeaderboard(results) {
    const container = document.getElementById('live-leaderboard-content');
    if (!container || !results || results.length === 0) return;
    // 按分数排序（简单解析 message 中的 score）
    const parsed = results.map((r, idx) => {
        const msg = r.message || '';
        const match = msg.match(/=([\d.]+)/);
        const score = match ? parseFloat(match[1]) : 0;
        const timeMatch = msg.match(/\(([\d.]+)s\)/);
        const timeVal = timeMatch ? timeMatch[1] + 's' : '';
        return { name: r.model_name || r.name || 'Model', score, timeVal, raw: msg };
    }).sort((a, b) => b.score - a.score);

    let html = '';
    parsed.forEach((item, idx) => {
        const rankClass = idx === 0 ? 'gold' : idx === 1 ? 'silver' : idx === 2 ? 'bronze' : '';
        html += `<div class="live-leaderboard-item">
            <div class="live-leaderboard-rank ${rankClass}">${idx + 1}</div>
            <div class="live-leaderboard-name">${escapeHtml(item.name)}</div>
            <div class="live-leaderboard-score">${item.score.toFixed(4)}</div>
            <div class="live-leaderboard-time">${item.timeVal}</div>
        </div>`;
    });
    container.innerHTML = html;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function switchMultiTargetTab(idx) {
    const buttons = document.querySelectorAll('#leaderboard-table button');
    buttons.forEach((btn, i) => {
        if (i === idx) {
            btn.style.background = 'var(--primary)';
            btn.style.color = '#fff';
        } else {
            btn.style.background = 'var(--surface)';
            btn.style.color = 'var(--text)';
        }
    });
    document.querySelectorAll('[id^="mt-panel-"]').forEach((panel, i) => {
        panel.style.display = i === idx ? 'block' : 'none';
    });
}

// ==================== 结果展示 ====================
async function loadTrainResult() {
    try {
        const res = await fetch('/api/model/result');
        const data = await res.json();
        if (!data.success) return;

        const result = data.result;

        // ===== 多目标结果展示 =====
        if (result.multi_target) {
            // 多目标：展示汇总
            document.getElementById('dec-confidence').textContent = `多目标训练`;
            let decHtml = `<h3 style="font-size:18px;margin-bottom:8px">多目标训练完成</h3>`;
            decHtml += `<p style="color:var(--text-light);margin-bottom:12px">共训练 ${result.targets.length} 个目标列，总耗时 ${result.total_time.toFixed(1)}s</p>`;
            decHtml += `<div class="config-display">`;
            result.leaderboards.forEach(item => {
                decHtml += `<div class="config-item"><div class="label">${item.target}</div><div class="value">${item.best_model || 'N/A'}</div></div>`;
            });
            decHtml += `</div>`;
            document.getElementById('dec-content').innerHTML = decHtml;
            document.getElementById('override-options').innerHTML = '<p style="color:#999">多目标模式暂不支持模型覆盖</p>';

            // 多目标排行榜：标签页形式
            if (result.leaderboards && result.leaderboards.length > 0) {
                let tabsHtml = '<div style="display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap;">';
                let panelsHtml = '';
                result.leaderboards.forEach((item, idx) => {
                    const active = idx === 0 ? 'background:var(--primary);color:#fff;' : 'background:var(--surface);color:var(--text);';
                    tabsHtml += `<button class="btn btn-sm" style="${active}" onclick="switchMultiTargetTab(${idx})">${item.target}</button>`;
                    const lb = item.leaderboard || [];
                    let tableHtml = '<p style="color:#999">无数据</p>';
                    if (lb.length > 0) {
                        const keys = Object.keys(lb[0]);
                        const thead = '<tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>';
                        const tbody = lb.map(row => '<tr>' + keys.map(k => `<td>${row[k] !== null ? (typeof row[k] === 'number' ? row[k].toFixed(4) : row[k]) : '-'}</td>`).join('') + '</tr>').join('');
                        tableHtml = `<table class="data-table"><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
                    }
                    const display = idx === 0 ? 'block' : 'none';
                    panelsHtml += `<div id="mt-panel-${idx}" style="display:${display}">${tableHtml}</div>`;
                });
                tabsHtml += '</div>';
                document.getElementById('leaderboard-table').innerHTML = tabsHtml + panelsHtml;
            }
        } else {
            // ===== 单目标结果展示（原有逻辑）=====
            // 决策报告
            if (result.decision && result.decision.recommended_model) {
                const dec = result.decision;
                document.getElementById('dec-confidence').textContent = `置信度 ${(dec.confidence * 100).toFixed(0)}%`;
                let decHtml = `
                    <h3 style="font-size:18px;margin-bottom:8px">${dec.recommended_name}</h3>
                    <p style="color:var(--text-light);margin-bottom:12px">${dec.recommendation_reason}</p>
                    <div class="config-display">
                        <div class="config-item"><div class="label">决策模式</div><div class="value">${dec.mode_description}</div></div>
                        <div class="config-item"><div class="label">置信度</div><div class="value">${(dec.confidence * 100).toFixed(0)}%</div></div>
                    </div>
                `;
                
                // 五维评分条形图
                if (dec.model_scores && dec.model_scores.length > 0) {
                    const rec = dec.model_scores.find(s => s.model_key === dec.recommended_model) || dec.model_scores[0];
                    const dims = [
                        { key: 'accuracy_score', label: '🎯 精度', color: 'linear-gradient(90deg,#3b82f6,#22c55e)' },
                        { key: 'speed_score', label: '⚡ 速度', color: 'linear-gradient(90deg,#f59e0b,#ef4444)' },
                        { key: 'stability_score', label: '🛡️ 稳定', color: 'linear-gradient(90deg,#8b5cf6,#a855f7)' },
                        { key: 'simplicity_score', label: '📦 简洁', color: 'linear-gradient(90deg,#06b6d4,#3b82f6)' },
                        { key: 'generalization_score', label: '🌍 泛化', color: 'linear-gradient(90deg,#10b981,#22c55e)' },
                    ];
                    const dimHtml = dims.map(d => {
                        const val = rec[d.key] || 0;
                        const pct = Math.min(100, Math.max(0, val));
                        return `<div class="dim-score-row"><span class="dim-score-label">${d.label}</span><div class="dim-score-bar-wrap"><div class="dim-score-bar" style="width:${pct}%;background:${d.color};"></div></div><span class="dim-score-val">${val.toFixed(1)}</span></div>`;
                    }).join('');
                    decHtml += `<div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--border);"><h4 style="font-size:13px;margin-bottom:8px;">五维能力评分</h4>${dimHtml}</div>`;
                }
                
                document.getElementById('dec-content').innerHTML = decHtml;

                // 覆盖选项
                const overrideDiv = document.getElementById('override-options');
                if (dec.override_options && Object.keys(dec.override_options).length > 0) {
                    overrideDiv.innerHTML = Object.entries(dec.override_options).map(([key, desc]) => `
                        <div class="override-option ${key === dec.recommended_model ? 'selected' : ''}" data-key="${key}" onclick="selectOverride(this)">
                            <input type="radio" name="override" value="${key}" ${key === dec.recommended_model ? 'checked' : ''}>
                            <div>
                                <strong>${key}</strong>
                                <div style="font-size:12px;color:var(--text-light)">${desc}</div>
                            </div>
                        </div>
                    `).join('');
                    selectedOverrideModel = dec.recommended_model;
                    // 默认显示导出卡片（推荐模型已确定）
                    const exportCard = document.getElementById('export-card');
                    if (exportCard) exportCard.classList.remove('hidden');
                } else {
                    overrideDiv.innerHTML = '<p style="color:#999">无可用覆盖选项</p>';
                }
            }

            // 排行榜
            if (result.leaderboard && result.leaderboard.length > 0) {
                const lb = result.leaderboard;
                const keys = Object.keys(lb[0]);
                const thead = '<tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>';
                const tbody = lb.map(row => '<tr>' + keys.map(k => `<td>${row[k] !== null ? (typeof row[k] === 'number' ? row[k].toFixed(4) : row[k]) : '-'}</td>`).join('') + '</tr>').join('');
                document.getElementById('leaderboard-table').innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
            }
        }

        // 预处理报告
        if (result.preprocessing) {
            const prep = result.preprocessing;
            const orig = prep.original_features || '-';
            const enc = prep.encoded_features || '-';
            const sel = prep.selected_features || '-';
            let flowHtml = `
                <div class="preprocessing-flow">
                    <div class="flow-step">
                        <div class="flow-badge">原始数据</div>
                        <div class="flow-detail">${orig} 特征</div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-step">
                        <div class="flow-badge">编码</div>
                        <div class="flow-detail">${enc} 特征</div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-step">
                        <div class="flow-badge">特征选择</div>
                        <div class="flow-detail">${sel} 特征</div>
                    </div>
                </div>
            `;
            if (prep.encoding_strategy) {
                flowHtml += `<div style="margin-top:8px;font-size:12px;color:var(--text-light)">编码策略: ${escapeHtml(prep.encoding_strategy)}</div>`;
            }
            document.getElementById('preprocessing-flow').innerHTML = flowHtml;
            
            if (result.encoding_report && result.encoding_report.length > 0) {
                const encDiv = document.getElementById('preprocessing-encoding');
                encDiv.classList.remove('hidden');
                encDiv.innerHTML = '<h4 style="font-size:13px;margin-bottom:6px;">编码详情</h4>' +
                    '<table class="data-table"><thead><tr><th>列</th><th>策略</th><th>输出列</th></tr></thead><tbody>' +
                    result.encoding_report.slice(0, 10).map(r => `<tr><td>${escapeHtml(r.column || r.col || '-')}</td><td>${escapeHtml(r.strategy || '-')}</td><td>${r.output_cols !== undefined ? r.output_cols : (r.new_columns !== undefined ? r.new_columns : '-')}</td></tr>`).join('') +
                    '</tbody></table>';
            }
            if (result.feature_selection_report && result.feature_selection_report.length > 0) {
                const fsDiv = document.getElementById('preprocessing-fs');
                fsDiv.classList.remove('hidden');
                fsDiv.innerHTML = '<h4 style="font-size:13px;margin-bottom:6px;">特征重要性 (Top 10)</h4>' +
                    '<table class="data-table"><thead><tr><th>特征</th><th>重要性</th></tr></thead><tbody>' +
                    result.feature_selection_report.slice(0, 10).map(r => `<tr><td>${escapeHtml(r.feature || r.column || '-')}</td><td>${(r.importance !== undefined ? r.importance : '-').toFixed ? r.importance.toFixed(4) : r.importance}</td></tr>`).join('') +
                    '</tbody></table>';
            }
        }

        // Permutation Importance
        if (result.permutation_importance && result.permutation_importance.length > 0) {
            document.getElementById('permutation-importance-card').classList.remove('hidden');
            renderPermutationImportance(result.permutation_importance);
        }
        
        // SHAP Explanation
        if (result.best_model && result.best_model.model_key) {
            loadSHAPExplanation(result.best_model.model_key);
        }
        
        // Robustness Test
        const robCard = document.getElementById('robustness-card');
        if (robCard) robCard.classList.remove('hidden');
        
        // Incremental Learning
        const incCard = document.getElementById('incremental-card');
        if (incCard) {
            if (result.best_model && (result.best_model.model_key === 'sgd' || result.best_model.model_key === 'passiveaggressive')) {
                incCard.classList.remove('hidden');
            } else {
                incCard.classList.add('hidden');
            }
        }
        
        // 伪标签报告
        if (result.pseudo_label_report) {
            const pr = result.pseudo_label_report;
            document.getElementById('pseudo-label-card').classList.remove('hidden');
            document.getElementById('pseudo-label-content').innerHTML = `
                <div class="config-display">
                    <div class="config-item"><div class="label">原始训练样本</div><div class="value">${pr.n_original}</div></div>
                    <div class="config-item"><div class="label">伪标签样本</div><div class="value" style="color:#10b981;font-weight:600;">+${pr.n_pseudo}</div></div>
                    <div class="config-item"><div class="label">增强后总样本</div><div class="value">${pr.n_combined}</div></div>
                    <div class="config-item"><div class="label">平均置信度</div><div class="value">${(pr.mean_confidence * 100).toFixed(1)}%</div></div>
                </div>
            `;
        }

        // AutoML 决策解释
        if (result.automl_decision) {
            const ad = result.automl_decision;
            document.getElementById('automl-decision-content').innerHTML = `
                <div class="config-display">
                    <div class="config-item"><div class="label">推荐优化器</div><div class="value">${ad.recommended_optimizer}</div></div>
                    <div class="config-item"><div class="label">推荐模型</div><div class="value">${ad.recommended_models.join(', ')}</div></div>
                    <div class="config-item"><div class="label">集成策略</div><div class="value">${ad.recommended_ensemble}</div></div>
                    <div class="config-item"><div class="label">预计耗时</div><div class="value">${ad.expected_time}</div></div>
                </div>
                <pre style="margin-top:12px;padding:10px;background:#f8f9fa;border-radius:6px;font-size:12px;white-space:pre-wrap;">${ad.meta_features}</pre>
            `;
            document.getElementById('automl-decision-card').classList.remove('hidden');
        }
        
        // 采样报告
        if (result.sampling && result.sampling.sample_ratio < 1.0) {
            const sr = result.sampling;
            document.getElementById('sampling-content').innerHTML = `
                <div class="config-display">
                    <div class="config-item"><div class="label">原始样本</div><div class="value">${sr.original_n.toLocaleString()}</div></div>
                    <div class="config-item"><div class="label">采样后</div><div class="value">${sr.sampled_n.toLocaleString()}</div></div>
                    <div class="config-item"><div class="label">采样比例</div><div class="value">${(sr.sample_ratio * 100).toFixed(1)}%</div></div>
                    <div class="config-item"><div class="label">策略</div><div class="value">${sr.strategy}</div></div>
                    <div class="config-item"><div class="label">分布保持</div><div class="value">${(sr.distribution_preservation * 100).toFixed(1)}%</div></div>
                </div>
            `;
            document.getElementById('sampling-report-card').classList.remove('hidden');
        }

        // 风险与建议
        if (result.decision && result.decision.risks && result.decision.risks.length > 0) {
            document.getElementById('risk-content').innerHTML = result.decision.risks.map(r =>
                `<div style="padding:8px 0;border-bottom:1px solid #eee">⚠️ ${r}</div>`
            ).join('') + `<div style="margin-top:12px;padding:10px;background:#e8f5e9;border-radius:6px">💡 ${result.decision.scenario_advice || '当前模型表现良好，可直接使用。'}</div>`;
        } else {
            document.getElementById('risk-content').innerHTML = '<div style="color:#28a745">✅ 无显著风险，模型表现良好。</div>';
        }

        // 模型排行榜
        if (result.leaderboard && result.leaderboard.length > 0) {
            renderModelLeaderboard(result.leaderboard, result.task_type, result.decision?.model_scores);
            renderModelComparison(result.leaderboard, result.task_type);
        }
        
        // 超参优化日志
        await loadHyperoptHistory();
        
        // 手动调参面板
        await initManualTunePanel();
        
        // 填充模型解释下拉框
        if (result.leaderboard && result.leaderboard.length > 0) {
            const select = document.getElementById('explain-model-select');
            select.innerHTML = '<option value="">-- 选择模型 --</option>' +
                result.leaderboard.map(m => `<option value="${m.model_key || m.model}">${m.model_name || m.model}</option>`).join('');
            // 默认选择最佳模型
            if (select.options.length > 1) {
                select.selectedIndex = 1;
                loadModelExplain();
            }
            
            // 填充公平性下拉框
            const fairnessSelect = document.getElementById('fairness-model-select');
            fairnessSelect.innerHTML = '<option value="">-- 选择模型 --</option>' +
                result.leaderboard.map(m => `<option value="${m.model_key || m.model}">${m.model_name || m.model}</option>`).join('');
            document.getElementById('fairness-card').classList.remove('hidden');
        }

    } catch (e) {
        showToast('加载结果失败', 'error');
    }
}

async function loadFairnessReport() {
    const modelKey = document.getElementById('fairness-model-select').value;
    const sensitiveAttr = document.getElementById('fairness-attr-input').value || null;
    const statusDiv = document.getElementById('fairness-status');
    const contentDiv = document.getElementById('fairness-content');
    
    if (!modelKey) {
        statusDiv.textContent = '请选择模型';
        return;
    }
    
    statusDiv.textContent = '正在分析公平性...';
    try {
        const res = await fetch('/api/model/fairness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: modelKey, sensitive_attr: sensitiveAttr })
        });
        const data = await res.json();
        if (!data.success) {
            statusDiv.textContent = '分析失败: ' + (data.error || '未知错误');
            contentDiv.classList.add('hidden');
            return;
        }
        
        const r = data.result;
        statusDiv.textContent = `敏感属性: ${r.sensitive_attr} | 分析耗时: ${(r.analysis_time || 0).toFixed(2)}s`;
        contentDiv.classList.remove('hidden');
        
        // 摘要
        const summaryDiv = document.getElementById('fairness-summary');
        const isFair = r.is_fair;
        summaryDiv.style.background = isFair ? '#e8f5e9' : '#ffebee';
        summaryDiv.innerHTML = `
            <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:13px;">
                <div><strong>Demographic Parity Diff:</strong> ${formatCell(r.demographic_parity_diff)}</div>
                <div><strong>Equalized Odds Diff:</strong> ${formatCell(r.equalized_odds_diff)}</div>
                <div><strong>FPR Diff:</strong> ${formatCell(r.fpr_diff)}</div>
                <div><strong>FNR Diff:</strong> ${formatCell(r.fnr_diff)}</div>
            </div>
            ${r.recommendations.length > 0 ? '<div style="margin-top:8px;color:#c62828;">' + r.recommendations.map(rec => '⚠️ ' + rec).join('<br>') + '</div>' : '<div style="margin-top:8px;color:#2e7d32;">✅ 模型通过公平性检查</div>'}
        `;
        
        // 群体指标表格
        const table = document.getElementById('fairness-table');
        if (r.group_metrics && Object.keys(r.group_metrics).length > 0) {
            const groups = Object.entries(r.group_metrics);
            const cols = ['group', 'count', 'accuracy', 'precision', 'recall', 'mse'];
            const thead = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
            const tbody = groups.map(([g, m]) => '<tr>' + cols.map(c => `<td>${formatCell(c === 'group' ? g : m[c])}</td>`).join('') + '</tr>').join('');
            table.innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
        } else {
            table.innerHTML = '<tbody><tr><td>无群体指标数据</td></tr></tbody>';
        }
    } catch (e) {
        statusDiv.textContent = '请求失败: ' + e.message;
        contentDiv.classList.add('hidden');
    }
}

async function loadModelExplain() {
    const modelKey = document.getElementById('explain-model-select').value;
    const instanceIdx = parseInt(document.getElementById('explain-instance-index').value, 10) || 0;
    const statusDiv = document.getElementById('explain-status');
    
    if (!modelKey) {
        statusDiv.textContent = '请选择模型';
        return;
    }
    
    statusDiv.textContent = '正在生成解释...';
    try {
        const res = await fetch('/api/model/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: modelKey, instance_index: instanceIdx })
        });
        const data = await res.json();
        if (!data.success) {
            statusDiv.textContent = '解释失败: ' + (data.error || '未知错误');
            return;
        }
        
        const r = data.result;
        statusDiv.textContent = `方法: ${r.method || 'builtin'} | 耗时: ${(r.explanation_time || 0).toFixed(2)}s`;
        
        // 全局重要性表格
        const globalSection = document.getElementById('explain-global-section');
        if (r.global_importance && r.global_importance.length > 0) {
            globalSection.classList.remove('hidden');
            const cols = Object.keys(r.global_importance[0]);
            const thead = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
            const tbody = r.global_importance.map(row => '<tr>' + cols.map(c => `<td>${formatCell(row[c])}</td>`).join('') + '</tr>').join('');
            document.getElementById('explain-global-table').innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
        } else {
            globalSection.classList.add('hidden');
        }
        
        // 单样本解释
        const instSection = document.getElementById('explain-instance-section');
        const inst = r.instance;
        if (inst) {
            instSection.classList.remove('hidden');
            let predHtml = '';
            if (inst.prediction) {
                if (inst.prediction.probability !== undefined) {
                    predHtml = `预测类别: <strong>${inst.prediction.class}</strong> (概率: ${(inst.prediction.probability * 100).toFixed(1)}%)`;
                } else if (inst.prediction.value !== undefined) {
                    predHtml = `预测值: <strong>${inst.prediction.value.toFixed(4)}</strong>`;
                }
            }
            document.getElementById('explain-prediction').innerHTML = predHtml;
            
            // 正负向特征贡献
            let contribRows = [];
            if (inst.top_positive && inst.top_positive.length > 0) {
                contribRows.push(...inst.top_positive.map(([f, v]) => `<tr><td>${f}</td><td style="color:#28a745">+${v.toFixed(4)}</td></tr>`));
            }
            if (inst.top_negative && inst.top_negative.length > 0) {
                contribRows.push(...inst.top_negative.map(([f, v]) => `<tr><td>${f}</td><td style="color:#dc3545">${v.toFixed(4)}</td></tr>`));
            }
            if (contribRows.length > 0) {
                document.getElementById('explain-instance-table').innerHTML = 
                    '<thead><tr><th>特征</th><th>贡献值</th></tr></thead><tbody>' + contribRows.join('') + '</tbody>';
            } else if (inst.lime && inst.lime.explanation) {
                const exp = inst.lime.explanation;
                const rows = Object.entries(exp).map(([f, v]) => `<tr><td>${f}</td><td>${typeof v === 'number' ? v.toFixed(4) : v}</td></tr>`);
                document.getElementById('explain-instance-table').innerHTML = 
                    '<thead><tr><th>特征</th><th>LIME 权重</th></tr></thead><tbody>' + rows.join('') + '</tbody>';
            } else {
                document.getElementById('explain-instance-table').innerHTML = '<tbody><tr><td>无可用的局部解释</td></tr></tbody>';
            }
        } else {
            instSection.classList.add('hidden');
        }
    } catch (e) {
        statusDiv.textContent = '请求失败: ' + e.message;
    }
}

function formatCell(v) {
    if (v === null || v === undefined) return '-';
    if (typeof v === 'number') return v.toFixed(4);
    return v;
}

async function loadHyperoptHistory() {
    try {
        const res = await fetch('/api/model/hyperopt-history');
        const data = await res.json();
        const card = document.getElementById('hyperopt-history-card');
        if (!data.success) {
            card.classList.add('hidden');
            return;
        }
        card.classList.remove('hidden');

        const models = data.models || [];
        document.getElementById('hyperopt-summary').textContent =
            `共 ${data.total_trials || 0} 次参数尝试，涉及 ${models.length} 个模型`;

        // 模型标签页
        const tabsDiv = document.getElementById('hyperopt-model-tabs');
        let activeModel = models.length > 0 ? models[0].model_key : '';
        
        tabsDiv.innerHTML = models.map((m, idx) => `
            <button class="btn btn-sm ${idx === 0 ? 'btn-primary' : 'btn-secondary'}" 
                    onclick="renderHyperoptTrials('${m.model_key}')" 
                    data-model="${m.model_key}">
                ${m.model_name} (${m.trial_count})
            </button>
        `).join('');

        // 保存到全局以便切换
        window._hyperoptData = models.reduce((acc, m) => { acc[m.model_key] = m; return acc; }, {});
        
        if (activeModel) {
            renderHyperoptTrials(activeModel);
        }
    } catch (e) {
        console.log('hyperopt history not available');
    }
}

function renderHyperoptTrials(modelKey) {
    const data = window._hyperoptData && window._hyperoptData[modelKey];
    if (!data) return;
    
    // 更新标签样式
    document.querySelectorAll('#hyperopt-model-tabs button').forEach(btn => {
        if (btn.dataset.model === modelKey) {
            btn.classList.remove('btn-secondary');
            btn.classList.add('btn-primary');
        } else {
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-secondary');
        }
    });
    
    const trials = data.trials || [];
    if (trials.length === 0) {
        document.getElementById('hyperopt-table').innerHTML = '<tr><td>无记录</td></tr>';
        return;
    }
    
    // 收集所有参数列
    const paramKeys = new Set();
    trials.forEach(t => {
        const p = t.params || t;
        Object.keys(p).forEach(k => {
            if (k !== 'number' && k !== 'value' && k !== 'trial' && k !== 'score' && k !== 'best_so_far' && k !== 'reward' && k !== 'epsilon') {
                paramKeys.add(k);
            }
        });
    });
    const pkeys = Array.from(paramKeys);
    
    // 渲染表格
    const thead = '<tr><th>#</th><th>Score</th>' + pkeys.map(k => `<th>${k}</th>`).join('') + '<th>Action</th></tr>';
    const tbody = trials.map((t, i) => {
        const scoreVal = t.score !== undefined ? t.score : (t.value !== undefined ? t.value : '-');
        const score = typeof scoreVal === 'number' ? scoreVal.toFixed(4) : scoreVal;
        const trialNum = t.trial !== undefined ? t.trial : (t.number !== undefined ? t.number + 1 : i + 1);
        const params = t.params || {};
        const pcells = pkeys.map(k => `<td>${params[k] !== undefined ? params[k] : '-'}</td>`).join('');
        const isBest = t.best_so_far !== undefined && Math.abs(scoreVal - t.best_so_far) < 1e-6;
        const highlight = isBest ? 'style="background:#e8f5e9;font-weight:600;"' : '';
        const paramsJson = JSON.stringify(params).replace(/"/g, '&quot;');
        return `<tr ${highlight}><td>${trialNum}</td><td>${score}</td>${pcells}<td><button class="btn btn-sm" onclick="reproduceHyperoptParams('${paramsJson}')">Reproduce</button></td></tr>`;
    }).join('');
    
    document.getElementById('hyperopt-table').innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
    
    // 渲染 ECharts 图表
    renderHyperoptCharts(modelKey, trials, pkeys);
}

function reproduceHyperoptParams(paramsJson) {
    try {
        const params = JSON.parse(paramsJson.replace(/&quot;/g, '"'));
        // 这里可以将参数加载到配置中，但超参通常是模型特定的
        // 简单显示参数供用户手动复制
        alert('Params: ' + JSON.stringify(params, null, 2));
    } catch (e) {
        showToast('Failed to parse params', 'error');
    }
}

function renderHyperoptCharts(modelKey, trials, pkeys) {
    // 1. Score 趋势图
    const scoreChartDom = document.getElementById('hyperopt-score-chart');
    if (scoreChartDom) {
        const existing = echarts.getInstanceByDom(scoreChartDom);
        if (existing) existing.dispose();
        const scoreChart = echarts.init(scoreChartDom);
        const xData = trials.map((t, i) => t.trial !== undefined ? t.trial : (t.number !== undefined ? t.number + 1 : i + 1));
        const yData = trials.map(t => t.score !== undefined ? t.score : (t.value !== undefined ? t.value : 0));
        const bestSoFar = [];
        let best = -Infinity;
        yData.forEach(v => { if (v > best) best = v; bestSoFar.push(best); });
        
        const yMin = Math.min(...yData), yMax = Math.max(...yData);
        const yPad = (yMax - yMin) * 0.1 || 1;
        
        scoreChart.setOption({
            tooltip: { trigger: 'axis' },
            legend: { data: ['Trial Score', 'Best So Far'], top: 0, textStyle: { fontSize: 11 } },
            grid: { left: 55, right: 20, top: 35, bottom: 25 },
            xAxis: { type: 'category', data: xData, name: 'Trial', nameTextStyle: { fontSize: 11 } },
            yAxis: { type: 'value', name: 'Score', nameTextStyle: { fontSize: 11 }, min: yMin - yPad, max: yMax + yPad },
            series: [
                { name: 'Trial Score', type: 'scatter', data: yData, symbolSize: 10, itemStyle: { color: '#3b82f6', opacity: 0.7 }, emphasis: { itemStyle: { borderColor: '#fff', borderWidth: 2 } } },
                { name: 'Best So Far', type: 'line', data: bestSoFar, smooth: false, lineStyle: { color: '#22c55e', width: 1.5, type: 'dashed' }, itemStyle: { color: '#22c55e' }, symbol: 'none' }
            ],
            animation: false
        });
        setTimeout(() => scoreChart.resize(), 50);
        window.addEventListener('resize', () => scoreChart.resize());
    }
    
    // 2. Slice Plot — 填充下拉框并渲染默认参数
    const sliceSelect = document.getElementById('hyperopt-slice-param');
    if (sliceSelect && pkeys.length > 0) {
        sliceSelect.innerHTML = '<option value="">-- 选择参数 --</option>' +
            pkeys.map(k => `<option value="${k}">${k}</option>`).join('');
        if (pkeys.length > 0) {
            sliceSelect.value = pkeys[0];
            renderHyperoptSliceChart();
        }
    }
    
    // 3. 平行坐标图
    renderHyperoptParallelChart(trials, pkeys);
    
    // 4. 3D 搜索空间
    renderHyperopt3DChart();
}

function renderHyperoptSliceChart() {
    const chartDom = document.getElementById('hyperopt-slice-chart');
    if (!chartDom) return;
    const existing = echarts.getInstanceByDom(chartDom);
    if (existing) existing.dispose();
    
    const paramKey = document.getElementById('hyperopt-slice-param').value;
    if (!paramKey || !window._hyperoptData) return;
    
    // 找到当前模型的 trials
    const activeBtn = document.querySelector('#hyperopt-model-tabs button.btn-primary');
    const modelKey = activeBtn ? activeBtn.dataset.model : '';
    const data = window._hyperoptData[modelKey];
    if (!data) return;
    const trials = data.trials || [];
    
    const vals = trials.map(t => (t.params || t)[paramKey]).filter(v => v !== undefined);
    const isNumeric = vals.length > 0 && vals.every(v => typeof v === 'number');
    const scores = trials.map(t => t.score !== undefined ? t.score : (t.value !== undefined ? t.value : 0));
    
    const chart = echarts.init(chartDom);
    
    if (isNumeric) {
        // 数值参数：散点 + 局部回归拟合
        const scatterData = trials.map((t, i) => [(t.params || t)[paramKey], scores[i]]);
        const sorted = [...scatterData].sort((a, b) => a[0] - b[0]);
        const wSize = Math.max(2, Math.floor(sorted.length / 4));
        const lineData = [];
        for (let i = 0; i < sorted.length; i++) {
            const s = Math.max(0, i - Math.floor(wSize / 2));
            const e = Math.min(sorted.length, s + wSize);
            const avg = sorted.slice(s, e).reduce((a, b) => a + b[1], 0) / (e - s);
            lineData.push([sorted[i][0], avg]);
        }
        chart.setOption({
            tooltip: { trigger: 'axis' },
            grid: { left: 45, right: 15, top: 10, bottom: 25 },
            xAxis: { type: 'value', name: paramKey, nameTextStyle: { fontSize: 10 } },
            yAxis: { type: 'value', name: 'Score', nameTextStyle: { fontSize: 10 } },
            series: [
                { type: 'scatter', data: scatterData, symbolSize: 8, itemStyle: { color: '#3b82f6', opacity: 0.6 } },
                { type: 'line', data: lineData, smooth: true, showSymbol: false, lineStyle: { color: '#f59e0b', width: 2, type: 'dashed' } }
            ],
            animation: false
        });
    } else {
        // 分类参数：箱线图
        const groups = {};
        trials.forEach((t, i) => {
            const v = String((t.params || t)[paramKey]);
            if (!groups[v]) groups[v] = [];
            groups[v].push(scores[i]);
        });
        const cats = Object.keys(groups);
        const boxData = cats.map(c => {
            const s = groups[c].sort((a, b) => a - b);
            const q1 = s[Math.floor(s.length * 0.25)];
            const q2 = s[Math.floor(s.length * 0.5)];
            const q3 = s[Math.floor(s.length * 0.75)];
            const min = s[0], max = s[s.length - 1];
            return [c, min, q1, q2, q3, max];
        });
        chart.setOption({
            tooltip: { trigger: 'item' },
            grid: { left: 45, right: 15, top: 10, bottom: 25 },
            xAxis: { type: 'category', data: cats, name: paramKey, nameTextStyle: { fontSize: 10 } },
            yAxis: { type: 'value', name: 'Score', nameTextStyle: { fontSize: 10 } },
            series: [{ type: 'boxplot', data: boxData.map(d => d.slice(1)), itemStyle: { color: '#93c5fd', borderColor: '#3b82f6' } }],
            animation: false
        });
    }
    setTimeout(() => chart.resize(), 50);
    window.addEventListener('resize', () => chart.resize());
}

function renderHyperoptParallelChart(trials, pkeys) {
    const chartDom = document.getElementById('hyperopt-parallel-chart');
    if (!chartDom || pkeys.length === 0 || trials.length === 0) return;
    const existing = echarts.getInstanceByDom(chartDom);
    if (existing) existing.dispose();
    
    // 确定数值型和分类型参数
    const numericKeys = pkeys.filter(k => {
        const vals = trials.map(t => (t.params || t)[k]).filter(v => v !== undefined);
        return vals.length > 0 && vals.every(v => typeof v === 'number');
    });
    const catKeys = pkeys.filter(k => !numericKeys.includes(k));
    
    // 最多显示 6 个维度（避免过度拥挤）
    const displayKeys = [...numericKeys, ...catKeys].slice(0, 6);
    if (displayKeys.length === 0) return;
    
    // 分类参数映射
    const catMaps = {};
    catKeys.forEach(k => {
        const vals = [...new Set(trials.map(t => (t.params || t)[k]).filter(v => v !== undefined))];
        catMaps[k] = vals;
    });
    
    const scores = trials.map(t => t.score !== undefined ? t.score : (t.value !== undefined ? t.value : 0));
    const minScore = Math.min(...scores), maxScore = Math.max(...scores);
    
    const parallelAxis = displayKeys.map(k => {
        if (catKeys.includes(k)) {
            return { dim: displayKeys.indexOf(k), name: k, type: 'category', data: catMaps[k] };
        }
        const vals = trials.map(t => (t.params || t)[k]).filter(v => v !== undefined);
        return { dim: displayKeys.indexOf(k), name: k, type: 'value', min: Math.min(...vals), max: Math.max(...vals) };
    });
    // score 作为最后一维
    parallelAxis.push({ dim: displayKeys.length, name: 'Score', type: 'value', min: minScore, max: maxScore });
    
    const lineData = trials.map((t, i) => {
        const row = displayKeys.map(k => {
            if (catKeys.includes(k)) return catMaps[k].indexOf((t.params || t)[k]);
            return (t.params || t)[k] ?? 0;
        });
        row.push(scores[i]);
        return row;
    });
    
    const chart = echarts.init(chartDom);
    chart.setOption({
        tooltip: { padding: 10, backgroundColor: '#222', borderColor: '#777', textStyle: { color: '#fff' } },
        parallelAxis: parallelAxis,
        parallel: { left: 80, right: 80, top: 30, bottom: 20, parallelAxisDefault: { type: 'value', nameLocation: 'end', nameGap: 15, nameTextStyle: { fontSize: 11 } } },
        visualMap: {
            min: minScore, max: maxScore, dimension: displayKeys.length,
            inRange: { color: ['#93c5fd', '#3b82f6', '#1e40af'] },
            right: 10, top: 30, itemHeight: 80, textStyle: { fontSize: 10 }
        },
        series: [{
            type: 'parallel',
            lineStyle: { width: 1.5, opacity: 0.4 },
            data: lineData,
            emphasis: { lineStyle: { width: 3, opacity: 0.8 } }
        }],
        animation: false
    });
    setTimeout(() => chart.resize(), 50);
    window.addEventListener('resize', () => chart.resize());
}

let _echartsGlLoaded = false;

function enableHyperopt3D() {
    const placeholder = document.getElementById('hyperopt-3d-placeholder');
    const controls = document.getElementById('hyperopt-3d-controls');
    const chartDiv = document.getElementById('hyperopt-3d-chart');
    
    if (_echartsGlLoaded) {
        placeholder.classList.add('hidden');
        controls.classList.remove('hidden');
        chartDiv.classList.remove('hidden');
        renderHyperopt3DChart();
        return;
    }
    
    // 动态加载 echarts-gl
    if (placeholder) {
        placeholder.innerHTML = '<div style="color:var(--text-light);font-size:13px;">⏳ 正在加载 ECharts GL...</div>';
    }
    
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/echarts-gl@2.0.9/dist/echarts-gl.min.js';
    script.onload = () => {
        _echartsGlLoaded = true;
        if (placeholder) placeholder.classList.add('hidden');
        if (controls) controls.classList.remove('hidden');
        if (chartDiv) chartDiv.classList.remove('hidden');
        renderHyperopt3DChart();
    };
    script.onerror = () => {
        if (placeholder) {
            placeholder.innerHTML = '<div style="color:#dc2626;font-size:13px;">❌ ECharts GL 加载失败<br/><span style="font-size:11px;color:var(--text-light);">请检查网络连接或浏览器 WebGL 支持</span></div>';
        }
    };
    document.head.appendChild(script);
}

function renderHyperopt3DChart() {
    const chartDom = document.getElementById('hyperopt-3d-chart');
    if (!chartDom || !window._hyperoptData) return;
    
    // 如果 3D 未开启，直接返回（占位按钮显示中）
    if (!chartDom.classList.contains('hidden') && !_echartsGlLoaded) {
        return;
    }
    
    const activeBtn = document.querySelector('#hyperopt-model-tabs button.btn-primary');
    const modelKey = activeBtn ? activeBtn.dataset.model : '';
    const data = window._hyperoptData[modelKey];
    if (!data) return;
    const trials = data.trials || [];
    if (trials.length === 0) return;
    
    // 提取数值参数
    const pkeys = data.pkeys || [];
    const numericKeys = pkeys.filter(k => {
        const vals = trials.map(t => (t.params || t)[k]).filter(v => v !== undefined);
        return vals.length > 0 && vals.every(v => typeof v === 'number');
    });
    
    if (numericKeys.length < 2) {
        chartDom.innerHTML = '<div style="text-align:center;padding:80px 0;color:var(--text-light);">Need at least 2 numeric parameters for 3D view</div>';
        return;
    }
    
    // 填充下拉框
    const xSelect = document.getElementById('hyperopt-3d-x');
    const ySelect = document.getElementById('hyperopt-3d-y');
    if (xSelect && xSelect.options.length <= 1) {
        const opts = numericKeys.map(k => `<option value="${k}">${k}</option>`).join('');
        xSelect.innerHTML = '<option value="">X axis</option>' + opts;
        ySelect.innerHTML = '<option value="">Y axis</option>' + opts;
        xSelect.value = numericKeys[0];
        ySelect.value = numericKeys[1] || numericKeys[0];
    }
    
    const xKey = xSelect ? xSelect.value : numericKeys[0];
    const yKey = ySelect ? ySelect.value : numericKeys[1];
    if (!xKey || !yKey || xKey === yKey) {
        chartDom.innerHTML = '<div style="text-align:center;padding:80px 0;color:var(--text-light);">Please select two different parameters</div>';
        return;
    }
    
    const scores = trials.map(t => t.score !== undefined ? t.score : (t.value !== undefined ? t.value : 0));
    const minScore = Math.min(...scores), maxScore = Math.max(...scores);
    
    const scatterData = trials.map((t, i) => {
        const x = (t.params || t)[xKey];
        const y = (t.params || t)[yKey];
        const z = scores[i];
        return [x, y, z, i + 1]; // [x, y, z, trialNumber]
    }).filter(d => d[0] !== undefined && d[1] !== undefined);
    
    if (scatterData.length === 0) return;
    
    const existing = echarts.getInstanceByDom(chartDom);
    if (existing) existing.dispose();
    
    try {
        const chart = echarts.init(chartDom);
        chart.setOption({
            tooltip: {
                formatter: (p) => {
                    const d = p.data;
                    return `Trial ${d[3]}<br/>${xKey}: ${d[0]}<br/>${yKey}: ${d[1]}<br/>Score: ${d[2].toFixed(4)}`;
                }
            },
            visualMap: {
                min: minScore, max: maxScore,
                inRange: { color: ['#93c5fd', '#3b82f6', '#1e40af', '#f59e0b', '#22c55e'] },
                right: 10, top: 10, itemHeight: 80, textStyle: { fontSize: 10 },
                calculable: true
            },
            xAxis3D: { name: xKey, nameTextStyle: { fontSize: 11 }, type: 'value' },
            yAxis3D: { name: yKey, nameTextStyle: { fontSize: 11 }, type: 'value' },
            zAxis3D: { name: 'Score', nameTextStyle: { fontSize: 11 }, type: 'value' },
            grid3D: {
                boxWidth: 200, boxDepth: 200, boxHeight: 120,
                viewControl: { projection: 'perspective', autoRotate: false, distance: 280 },
                light: { main: { intensity: 1.2, shadow: true }, ambient: { intensity: 0.3 } }
            },
            series: [{
                type: 'scatter3D',
                data: scatterData,
                symbolSize: 10,
                emphasis: { itemStyle: { borderColor: '#fff', borderWidth: 2 } }
            }],
            animation: false
        });
        setTimeout(() => chart.resize(), 50);
        window.addEventListener('resize', () => chart.resize());
        
        const hint = document.getElementById('hyperopt-3d-hint');
        if (hint) hint.textContent = `${scatterData.length} trials · drag to rotate · scroll to zoom`;
    } catch (e) {
        chartDom.innerHTML = `<div style="text-align:center;padding:80px 0;color:var(--text-light);">3D chart failed to load.<br/><small>${e.message}</small></div>`;
    }
}

function autoSelect3DAxes() {
    if (!window._hyperoptData) return;
    const activeBtn = document.querySelector('#hyperopt-model-tabs button.btn-primary');
    const modelKey = activeBtn ? activeBtn.dataset.model : '';
    const data = window._hyperoptData[modelKey];
    if (!data) return;
    const trials = data.trials || [];
    const pkeys = data.pkeys || [];
    
    const numericKeys = pkeys.filter(k => {
        const vals = trials.map(t => (t.params || t)[k]).filter(v => v !== undefined);
        return vals.length > 0 && vals.every(v => typeof v === 'number');
    });
    
    if (numericKeys.length < 2) return;
    
    // 选方差最大的两个参数
    const variances = numericKeys.map(k => {
        const vals = trials.map(t => (t.params || t)[k]).filter(v => v !== undefined);
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const var_ = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
        return { key: k, var: var_ };
    }).sort((a, b) => b.var - a.var);
    
    const xSelect = document.getElementById('hyperopt-3d-x');
    const ySelect = document.getElementById('hyperopt-3d-y');
    if (xSelect && variances[0]) xSelect.value = variances[0].key;
    if (ySelect && variances[1]) ySelect.value = variances[1].key;
    
    renderHyperopt3DChart();
}

function selectOverride(el) {
    document.querySelectorAll('.override-option').forEach(o => o.classList.remove('selected'));
    el.classList.add('selected');
    el.querySelector('input').checked = true;
    selectedOverrideModel = el.dataset.key;
}

function confirmOverride() {
    if (!selectedOverrideModel) {
        showToast('请先选择一个模型', 'error');
        return;
    }
    showToast(`已选择模型: ${selectedOverrideModel}，可导出摘要与代码`);
    // 显示导出卡片
    const exportCard = document.getElementById('export-card');
    if (exportCard) exportCard.classList.remove('hidden');
}

// ==================== 模型导出 ====================
let _exportData = { summary: '', code: '' };
let _exportTab = 'summary';

async function exportModel(format) {
    const modelKey = selectedOverrideModel;
    if (!modelKey) {
        showToast('请先选择模型', 'error');
        return;
    }
    
    try {
        const res = await fetch('/api/model/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: modelKey, format: format })
        });
        const data = await res.json();
        
        if (data.success) {
            if (format === 'summary' || format === 'both') {
                _exportData.summary = data.summary || '';
            }
            if (format === 'code' || format === 'both') {
                _exportData.code = data.code || '';
            }
            _exportTab = format === 'code' ? 'code' : 'summary';
            openExportModal();
        } else {
            showToast(data.error || '导出失败', 'error');
        }
    } catch (e) {
        showToast('导出请求失败: ' + e.message, 'error');
    }
}

function openExportModal() {
    const modal = document.getElementById('export-modal');
    if (modal) modal.classList.remove('hidden');
    switchExportTab(_exportTab);
}

function closeExportModal() {
    const modal = document.getElementById('export-modal');
    if (modal) modal.classList.add('hidden');
}

function switchExportTab(tab) {
    _exportTab = tab;
    document.querySelectorAll('#export-modal .modal-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    const content = document.getElementById('export-content');
    const title = document.getElementById('export-modal-title');
    if (content) {
        content.textContent = tab === 'summary' ? _exportData.summary : _exportData.code;
    }
    if (title) {
        title.textContent = tab === 'summary' ? '📋 模型摘要' : '💾 可复用代码';
    }
}

async function copyExport() {
    const text = _exportTab === 'summary' ? _exportData.summary : _exportData.code;
    try {
        await navigator.clipboard.writeText(text);
        showToast('已复制到剪贴板');
    } catch (e) {
        // fallback
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast('已复制到剪贴板');
    }
}

// ==================== 实验历史 ====================
let _experimentsData = [];

function toggleExperimentsPanel() {
    const card = document.getElementById('experiments-card');
    if (!card) return;
    const isHidden = card.classList.contains('hidden');
    if (isHidden) {
        card.classList.remove('hidden');
        loadExperiments();
    } else {
        card.classList.add('hidden');
    }
}

async function loadExperiments() {
    const tbody = document.getElementById('experiments-tbody');
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-light);">Loading...</td></tr>';
    try {
        const res = await fetch('/api/experiments');
        const data = await res.json();
        if (data.success) {
            _experimentsData = data.experiments || [];
            renderExperiments();
        } else {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#e74c3c;">Failed to load</td></tr>';
        }
    } catch (e) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#e74c3c;">Error: ' + e.message + '</td></tr>';
    }
}

function renderExperiments() {
    const tbody = document.getElementById('experiments-tbody');
    if (!tbody) return;
    if (_experimentsData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-light);">No experiments yet.</td></tr>';
        return;
    }
    const rows = _experimentsData.map((exp, idx) => {
        const timeStr = exp.timestamp || 'N/A';
        const modelName = exp.best_model || 'N/A';
        const score = exp.best_score !== undefined && exp.best_score !== null ? exp.best_score.toFixed(4) : 'N/A';
        const time = exp.duration !== undefined && exp.duration !== null ? exp.duration.toFixed(1) + 's' : 'N/A';
        const task = exp.task_type || 'N/A';
        const expId = exp.id;
        return '<tr>' +
            '<td><input type="checkbox" class="exp-select" data-id="' + expId + '" data-idx="' + idx + '"></td>' +
            '<td>' + expId + '</td>' +
            '<td>' + timeStr + '</td>' +
            '<td>' + task + '</td>' +
            '<td>' + modelName + '</td>' +
            '<td>' + score + '</td>' +
            '<td>' + time + '</td>' +
            '<td>' +
            '<button class="btn btn-sm" onclick="showExperimentDetail(' + expId + ')" title="View detail">Detail</button> ' +
            '<button class="btn btn-sm" onclick="reproduceExperiment(' + idx + ')" title="Load config">Reproduce</button> ' +
            '<button class="btn btn-sm" onclick="exportExperimentConfig(' + expId + ')" title="Export JSON">Export</button> ' +
            '<button class="btn btn-sm btn-secondary" onclick="deleteExperiment(' + expId + ')" title="Delete">Del</button>' +
            '</td>' +
            '</tr>';
    }).join('');
    tbody.innerHTML = rows;
}

function toggleSelectAllExperiments() {
    const master = document.getElementById('exp-select-all');
    const boxes = document.querySelectorAll('.exp-select');
    boxes.forEach(b => b.checked = master.checked);
}

function getSelectedExperimentIds() {
    const boxes = document.querySelectorAll('.exp-select:checked');
    return Array.from(boxes).map(b => parseInt(b.getAttribute('data-id')));
}

async function compareSelectedExperiments() {
    const ids = getSelectedExperimentIds();
    if (ids.length < 2) {
        showToast('Please select at least 2 experiments to compare', 'error');
        return;
    }
    try {
        const res = await fetch('/api/experiments/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids: ids })
        });
        const data = await res.json();
        if (data.success && data.comparison) {
            renderExperimentCompareModal(data.comparison);
        } else {
            showToast(data.error || 'Compare failed', 'error');
        }
    } catch (e) {
        showToast('Compare error: ' + e.message, 'error');
    }
}

function renderExperimentCompareModal(comparison) {
    const container = document.getElementById('experiment-compare-content');
    if (!container) return;
    // Determine best score direction: classification -> higher is better; regression -> lower is better (assuming RMSE/MAE)
    const task = comparison[0] && comparison[0].task_type ? comparison[0].task_type : '';
    const isHigherBetter = task === 'classification' || task === 'clustering';
    const scores = comparison.map(c => c.best_score).filter(v => v !== undefined && v !== null);
    const durs = comparison.map(c => c.duration).filter(v => v !== undefined && v !== null);
    const bestScore = isHigherBetter ? Math.max(...scores) : Math.min(...scores);
    const bestDur = Math.min(...durs);

    const headers = ['ID', 'Task', 'Best Model', 'Score', 'Duration', 'Config'];
    const rows = comparison.map(c => {
        const score = c.best_score !== undefined && c.best_score !== null ? c.best_score.toFixed(4) : 'N/A';
        const dur = c.duration !== undefined && c.duration !== null ? c.duration.toFixed(1) + 's' : 'N/A';
        const scoreClass = (c.best_score === bestScore) ? ' style="background:#d4edda;font-weight:bold;"' : '';
        const durClass = (c.duration === bestDur) ? ' style="background:#d4edda;font-weight:bold;"' : '';
        return '<tr><td>' + c.id + '</td><td>' + (c.task_type || 'N/A') + '</td><td>' + (c.best_model || 'N/A') + '</td><td' + scoreClass + '>' + score + '</td><td' + durClass + '>' + dur + '</td><td style="font-size:12px;">' + (c.config_summary || '') + '</td></tr>';
    }).join('');
    container.innerHTML = '<table class="data-table"><thead><tr>' + headers.map(h => '<th>' + h + '</th>').join('') + '</tr></thead><tbody>' + rows + '</tbody></table>';
    const modal = document.getElementById('experiment-compare-modal');
    if (modal) modal.classList.remove('hidden');
}

function showExperimentDetail(expId) {
    fetch('/api/experiments/' + expId)
        .then(r => r.json())
        .then(data => {
            if (!data.success || !data.experiment) {
                showToast('Failed to load detail', 'error');
                return;
            }
            const exp = data.experiment;
            const content = document.getElementById('experiment-detail-content');
            if (content) {
                content.textContent = JSON.stringify(exp, null, 2);
            }
            const modal = document.getElementById('experiment-detail-modal');
            if (modal) modal.classList.remove('hidden');
        })
        .catch(e => showToast('Error: ' + e.message, 'error'));
}

function closeExperimentDetailModal(e) {
    if (e && e.target !== e.currentTarget) return;
    const modal = document.getElementById('experiment-detail-modal');
    if (modal) modal.classList.add('hidden');
}

function closeExperimentCompareModal(e) {
    if (e && e.target !== e.currentTarget) return;
    const modal = document.getElementById('experiment-compare-modal');
    if (modal) modal.classList.add('hidden');
}

function exportExperimentConfig(expId) {
    fetch('/api/experiments/' + expId)
        .then(r => r.json())
        .then(data => {
            if (!data.success || !data.experiment) {
                showToast('Failed to load config', 'error');
                return;
            }
            const cfg = data.experiment.config || {};
            const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'experiment_' + expId + '_config.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showToast('Config exported', 'success');
        })
        .catch(e => showToast('Export error: ' + e.message, 'error'));
}

async function deleteExperiment(expId) {
    if (!confirm('Delete experiment #' + expId + '?')) return;
    try {
        const res = await fetch('/api/experiments/' + expId, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast('Deleted experiment #' + expId, 'success');
            loadExperiments();
        } else {
            showToast(data.error || 'Delete failed', 'error');
        }
    } catch (e) {
        showToast('Delete error: ' + e.message, 'error');
    }
}

function reproduceExperiment(idx) {
    const exp = _experimentsData[idx];
    if (!exp) {
        showToast('Experiment not available', 'error');
        return;
    }
    // Fetch full config from detail API
    fetch('/api/experiments/' + exp.id)
        .then(r => r.json())
        .then(data => {
            if (!data.success || !data.experiment || !data.experiment.config) {
                showToast('Config not available', 'error');
                return;
            }
            const cfg = data.experiment.config;
            if (cfg.task_type !== undefined) setSelectValue('cfg-task', cfg.task_type);
            if (cfg.ensemble !== undefined) setSelectValue('cfg-ensemble', cfg.ensemble);
            if (cfg.feature_engineering !== undefined) setSelectValue('cfg-feature-engineering', String(cfg.feature_engineering));
            if (cfg.fold_type !== undefined) setSelectValue('cfg-fold-type', cfg.fold_type);
            if (cfg.pseudo_labeling !== undefined) setSelectValue('cfg-pseudo-labeling', String(cfg.pseudo_labeling));
            if (cfg.models !== undefined) {
                const el = document.getElementById('cfg-models');
                if (el) el.value = cfg.models;
            }
            showToast('Configuration loaded. Go to Step 4 to review and train.', 'success');
        })
        .catch(e => showToast('Error loading config: ' + e.message, 'error'));
}

function setSelectValue(id, value) {
    const el = document.getElementById(id);
    if (el) {
        el.value = value;
    }
}

// ==================== 可视化 ====================
async function loadChart(chartType) {
    const img = document.getElementById('chart-image');
    const placeholder = document.getElementById('chart-placeholder');
    img.classList.add('hidden');
    placeholder.textContent = '正在生成图表...';
    placeholder.classList.remove('hidden');

    try {
        const res = await fetch('/api/visualization/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chart_type: chartType })
        });
        const data = await res.json();
        if (data.success) {
            img.src = data.image;
            img.classList.remove('hidden');
            placeholder.classList.add('hidden');
        } else {
            placeholder.textContent = '图表生成失败: ' + (data.error || '未知错误');
        }
    } catch (e) {
        placeholder.textContent = '请求失败: ' + e.message;
    }
}

// ==================== 预测 ====================
function initPredictUpload() {
    const input = document.getElementById('predict-input');
    input.addEventListener('change', e => {
        if (e.target.files.length) handlePredictUpload(e.target.files[0]);
    });
}

async function handlePredictUpload(file) {
    showToast(`正在预测: ${file.name}`);
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
            const preview = data.preview;
            const keys = Object.keys(preview[0]);
            const thead = '<tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>';
            const tbody = preview.map(row => '<tr>' + keys.map(k => `<td>${row[k] !== null ? row[k] : '-'}</td>`).join('') + '</tr>').join('');
            document.getElementById('predict-table').innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
            document.getElementById('predict-result').classList.remove('hidden');
            showToast('预测完成！');
        } else {
            showToast(data.error || '预测失败', 'error');
        }
    } catch (e) {
        showToast('预测出错: ' + e.message, 'error');
    }
}

// ==================== 工具函数 ====================
function showToast(msg, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = 'toast show';
    if (type === 'error') toast.style.background = 'var(--danger)';
    else if (type === 'success') toast.style.background = 'var(--success)';
    else toast.style.background = 'var(--text)';

    setTimeout(() => { toast.className = 'toast hidden'; }, 3000);
}

function resetAll() {
    if (confirm('确定要重置所有数据吗？')) {
        clearInterval(trainTimer);
        clearInterval(llmTimer);
        uploadedData = null;
        currentStep = 1;
        selectedOverrideModel = null;
        selectedAnalysisType = null;
        location.reload();
    }
}

// ==================== LLM 智能分析 ====================

function initLLMStep() {
    // 检查可用性并禁用不可用的分析类型
    checkLLMAnalysisAvailability();
    // 加载默认配置
    loadLLMDefaults();
}

async function loadLLMDefaults() {
    try {
        const res = await fetch('/api/llm/config');
        const data = await res.json();
        if (data.success && data.providers) {
            window.llmProviders = data.providers;
        }
    } catch (e) {
        console.error('加载 LLM 配置失败', e);
    }
}

function onLLMProviderChange() {
    const provider = document.getElementById('llm-provider').value;
    const providers = window.llmProviders || {};
    const cfg = providers[provider];
    if (cfg) {
        document.getElementById('llm-base-url').value = cfg.base_url;
        document.getElementById('llm-model-name').value = cfg.model_name;
        const keyGroup = document.getElementById('llm-api-key-group');
        if (cfg.needs_api_key) {
            keyGroup.classList.remove('hidden');
        } else {
            keyGroup.classList.add('hidden');
        }
    }
    // 控制浏览按钮显示：只有ollama模式显示
    const browseBtn = document.getElementById('llm-browse-btn');
    if (browseBtn) {
        if (provider === 'ollama') {
            browseBtn.classList.remove('hidden');
        } else {
            browseBtn.classList.add('hidden');
        }
    }
}

/* ==================== 本地模型浏览器 ==================== */

function openModelBrowser() {
    document.getElementById('model-browser-overlay').classList.remove('hidden');
}

function closeModelBrowser(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById('model-browser-overlay').classList.add('hidden');
}

function switchModelTab(tab) {
    document.querySelectorAll('.model-browser-tabs .tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === 'tab-' + tab);
    });
}

async function refreshOllamaModels() {
    const baseUrl = document.getElementById('llm-base-url').value;
    const listEl = document.getElementById('ollama-model-list');
    const hintEl = document.getElementById('ollama-hint');
    
    listEl.innerHTML = '<div class="model-list-empty">⏳ 正在获取模型列表...</div>';
    hintEl.textContent = '连接中...';
    
    try {
        // 提取Ollama基础URL（去掉/v1后缀）
        let ollamaBase = baseUrl;
        if (ollamaBase.endsWith('/v1')) {
            ollamaBase = ollamaBase.slice(0, -3);
        }
        
        const res = await fetch('/api/ollama/models?base_url=' + encodeURIComponent(ollamaBase));
        const data = await res.json();
        
        if (data.success) {
            hintEl.textContent = `共 ${data.models.length} 个模型`;
            renderModelList('ollama-model-list', data.models, 'ollama');
        } else {
            hintEl.textContent = '获取失败';
            listEl.innerHTML = `<div class="model-list-empty" style="color:#c00">❌ ${data.error}</div>`;
        }
    } catch (e) {
        hintEl.textContent = '请求失败';
        listEl.innerHTML = `<div class="model-list-empty" style="color:#c00">❌ 请求失败: ${e.message}</div>`;
    }
}

async function scanLocalFolder() {
    const path = document.getElementById('local-model-path').value.trim();
    const listEl = document.getElementById('folder-model-list');
    
    if (!path) {
        listEl.innerHTML = '<div class="model-list-empty" style="color:#c00">请输入文件夹路径</div>';
        return;
    }
    
    listEl.innerHTML = '<div class="model-list-empty">⏳ 正在扫描...</div>';
    
    try {
        const res = await fetch('/api/local-models/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: path })
        });
        const data = await res.json();
        
        if (data.success) {
            renderFolderModelList(data.models);
        } else {
            listEl.innerHTML = `<div class="model-list-empty" style="color:#c00">❌ ${data.error}</div>`;
        }
    } catch (e) {
        listEl.innerHTML = `<div class="model-list-empty" style="color:#c00">❌ 请求失败: ${e.message}</div>`;
    }
}

function renderModelList(containerId, models, source) {
    const container = document.getElementById(containerId);
    if (!models || models.length === 0) {
        container.innerHTML = '<div class="model-list-empty">未找到模型</div>';
        return;
    }
    
    let html = '';
    models.forEach(name => {
        html += `
            <div class="model-item" onclick="selectModel('${escapeHtml(name)}')">
                <div class="model-item-info">
                    <span class="model-item-name">${escapeHtml(name)}</span>
                </div>
                <span class="model-item-badge ${source}">${source === 'ollama' ? 'Ollama' : '本地'}</span>
            </div>
        `;
    });
    container.innerHTML = html;
}

function renderFolderModelList(models) {
    const container = document.getElementById('folder-model-list');
    if (!models || models.length === 0) {
        container.innerHTML = '<div class="model-list-empty">该文件夹未找到模型文件（支持 .gguf / .bin / .safetensors）</div>';
        return;
    }
    
    let html = '';
    models.forEach(m => {
        const name = m.name || m.filename;
        const size = m.size_mb ? `${m.size_mb} MB` : '';
        html += `
            <div class="model-item" onclick="selectModel('${escapeHtml(name)}')">
                <div class="model-item-info">
                    <span class="model-item-name">${escapeHtml(name)}</span>
                    <span class="model-item-meta">${escapeHtml(m.filename)} ${size ? '· ' + size : ''}</span>
                </div>
                <span class="model-item-badge">本地</span>
            </div>
        `;
    });
    container.innerHTML = html;
}

function selectModel(name) {
    document.getElementById('llm-model-name').value = name;
    closeModelBrowser();
    showToast('已选择模型: ' + name, 'success');
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function checkLLMAnalysisAvailability() {
    // 这些状态会在训练/上传过程中被设置，
    // 这里简单地让后端在 /api/llm/analyze 时做校验，
    // 前端只做视觉提示
    // 实际上我们在 startLLMAnalysis 前会再次检查
}

function selectAnalysisType(type) {
    selectedAnalysisType = type;
    document.querySelectorAll('.analysis-card').forEach(el => el.classList.remove('selected'));
    document.getElementById('card-' + type).classList.add('selected');
}

async function startLLMAnalysis() {
    if (!selectedAnalysisType) {
        showToast('请先选择一种分析类型', 'error');
        return;
    }

    const provider = document.getElementById('llm-provider').value;
    const baseUrl = document.getElementById('llm-base-url').value;
    const apiKey = document.getElementById('llm-api-key').value;
    const modelName = document.getElementById('llm-model-name').value;

    document.getElementById('llm-analyze-btn').disabled = true;
    document.getElementById('llm-status-text').textContent = '分析中...';
    document.getElementById('llm-result-card').classList.add('hidden');

    try {
        const res = await fetch('/api/llm/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                analysis_type: selectedAnalysisType,
                provider: provider,
                base_url: baseUrl,
                api_key: apiKey,
                model_name: modelName,
            })
        });
        const data = await res.json();
        if (data.success) {
            pollLLMStatus();
        } else {
            showToast(data.error || '分析启动失败', 'error');
            document.getElementById('llm-analyze-btn').disabled = false;
            document.getElementById('llm-status-text').textContent = '';
        }
    } catch (e) {
        showToast('分析请求失败: ' + e.message, 'error');
        document.getElementById('llm-analyze-btn').disabled = false;
        document.getElementById('llm-status-text').textContent = '';
    }
}

function pollLLMStatus() {
    if (llmTimer) clearInterval(llmTimer);
    llmTimer = setInterval(async () => {
        try {
            const res = await fetch('/api/llm/status');
            const data = await res.json();
            if (!data.success) return;

            if (data.status === 'done') {
                clearInterval(llmTimer);
                document.getElementById('llm-analyze-btn').disabled = false;
                document.getElementById('llm-status-text').textContent = '';
                renderLLMResult(data.result);
                showToast('AI 分析完成！');
            } else if (data.status === 'error') {
                clearInterval(llmTimer);
                document.getElementById('llm-analyze-btn').disabled = false;
                document.getElementById('llm-status-text').textContent = '';
                showToast('分析失败: ' + (data.error || '未知错误'), 'error');
            } else {
                document.getElementById('llm-status-text').textContent = '分析中，请稍候...';
            }
        } catch (e) {
            console.error(e);
        }
    }, 2000);
}

function renderLLMResult(text) {
    const container = document.getElementById('llm-result-content');
    container.innerHTML = renderMarkdown(text || '无分析结果');
    document.getElementById('llm-result-card').classList.remove('hidden');
}

// ==================== 报表设计器 ====================

let reportMode = 'pivot';
let reportFields = [];
let pivotConfig = {
    row_fields: [],
    col_fields: [],
    value_fields: [],
    aggregations: {},
    filters: {},
};
let freeCells = [];
let selectedCell = null;

function initReportDesigner() {
    loadReportFields();
    renderPivotZones();
    renderFreeCanvas();
    initZoomWheel();
}

async function loadReportFields() {
    const container = document.getElementById('field-list');
    container.innerHTML = '<div class="loading">加载字段中...</div>';
    
    try {
        const res = await fetch('/api/report/fields');
        const data = await res.json();
        if (data.success) {
            reportFields = data.fields;
            renderFieldList();
        } else {
            container.innerHTML = '<div class="loading">' + (data.error || '加载失败') + '</div>';
        }
    } catch (e) {
        container.innerHTML = '<div class="loading">加载失败</div>';
    }
}

function renderFieldList() {
    const container = document.getElementById('field-list');
    const icons = { numeric: '📊', text: '🏷️', datetime: '📅', boolean: '✅' };
    container.innerHTML = reportFields.map(f => `
        <div class="field-item" draggable="true"
             ondragstart="onFieldDragStart(event, '${f.name}')">
            <span class="field-icon">${icons[f.type] || '📋'}</span>
            <span class="field-name">${f.name}</span>
            <span class="field-type">${f.type}</span>
        </div>
    `).join('');
}

function onFieldDragStart(e, fieldName) {
    e.dataTransfer.setData('text/plain', fieldName);
    e.dataTransfer.effectAllowed = 'copy';
}

function onDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    e.currentTarget.classList.add('drag-over');
}

function onDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function onDropZone(e, zoneType) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    const fieldName = e.dataTransfer.getData('text/plain');
    if (!fieldName) return;
    
    if (zoneType === 'row_fields') {
        if (!pivotConfig.row_fields.includes(fieldName)) {
            pivotConfig.row_fields.push(fieldName);
        }
    } else if (zoneType === 'col_fields') {
        if (!pivotConfig.col_fields.includes(fieldName)) {
            pivotConfig.col_fields.push(fieldName);
        }
    } else if (zoneType === 'value_fields') {
        if (!pivotConfig.value_fields.includes(fieldName)) {
            pivotConfig.value_fields.push(fieldName);
            pivotConfig.aggregations[fieldName] = 'sum';
        }
    }
    renderPivotZones();
}

function removePivotField(zoneType, fieldName) {
    if (zoneType === 'row') {
        pivotConfig.row_fields = pivotConfig.row_fields.filter(f => f !== fieldName);
    } else if (zoneType === 'col') {
        pivotConfig.col_fields = pivotConfig.col_fields.filter(f => f !== fieldName);
    } else if (zoneType === 'value') {
        pivotConfig.value_fields = pivotConfig.value_fields.filter(f => f !== fieldName);
        delete pivotConfig.aggregations[fieldName];
    }
    renderPivotZones();
}

function renderPivotZones() {
    const renderItems = (fields, zoneType) => fields.map(f => {
        const agg = pivotConfig.aggregations[f] || '';
        const aggLabel = agg ? `(${agg})` : '';
        return `<div class="zone-item" onclick="editValueField('${f}')">
            ${f}${aggLabel}
            <span class="remove" onclick="event.stopPropagation(); removePivotField('${zoneType}', '${f}')">✕</span>
        </div>`;
    }).join('');
    
    document.getElementById('row-items').innerHTML = renderItems(pivotConfig.row_fields, 'row');
    document.getElementById('col-items').innerHTML = renderItems(pivotConfig.col_fields, 'col');
    document.getElementById('value-items').innerHTML = renderItems(pivotConfig.value_fields, 'value');
}

function editValueField(fieldName) {
    const aggOptions = [
        {value: 'sum', label: '求和'},
        {value: 'mean', label: '平均'},
        {value: 'count', label: '计数'},
        {value: 'max', label: '最大'},
        {value: 'min', label: '最小'},
        {value: 'std', label: '标准差'},
        {value: 'median', label: '中位数'},
    ];
    const formatOptions = [
        {value: '', label: '默认'},
        {value: '#,##0', label: '千分位整数'},
        {value: '#,##0.00', label: '千分位两位小数'},
        {value: '0%', label: '百分比'},
        {value: '0.00%', label: '百分比两位小数'},
    ];
    
    const currentAgg = pivotConfig.aggregations[fieldName] || 'sum';
    
    document.getElementById('property-content').innerHTML = `
        <div class="property-group">
            <label>字段</label>
            <input type="text" value="${fieldName}" disabled>
        </div>
        <div class="property-group">
            <label>聚合方式</label>
            <select id="prop-agg" onchange="updateValueAgg('${fieldName}', this.value)">
                ${aggOptions.map(o => `<option value="${o.value}" ${o.value === currentAgg ? 'selected' : ''}>${o.label}</option>`).join('')}
            </select>
        </div>
    `;
}

function updateValueAgg(fieldName, agg) {
    pivotConfig.aggregations[fieldName] = agg;
    renderPivotZones();
}

function switchReportMode(mode) {
    reportMode = mode;
    document.getElementById('tab-pivot').classList.toggle('active', mode === 'pivot');
    document.getElementById('tab-free').classList.toggle('active', mode === 'free');
    document.getElementById('pivot-panel').classList.toggle('hidden', mode !== 'pivot');
    document.getElementById('free-panel').classList.toggle('hidden', mode !== 'free');
}

// 自由报表模式
function renderFreeCanvas() {
    const table = document.getElementById('free-canvas');
    const rows = 12;
    const cols = 8;
    let html = '';
    for (let r = 0; r < rows; r++) {
        html += '<tr>';
        for (let c = 0; c < cols; c++) {
            html += `<td onclick="selectFreeCell(${r}, ${c})" data-row="${r}" data-col="${c}"></td>`;
        }
        html += '</tr>';
    }
    table.innerHTML = html;
}

function selectFreeCell(row, col) {
    selectedCell = { row, col };
    document.querySelectorAll('.free-canvas td').forEach(td => td.classList.remove('selected'));
    const td = document.querySelector(`.free-canvas td[data-row="${row}"][data-col="${col}"]`);
    if (td) td.classList.add('selected');
    
    showFreeCellProperties();
}

function showFreeCellProperties() {
    if (!selectedCell) return;
    
    document.getElementById('property-content').innerHTML = `
        <div class="property-group">
            <label>单元格类型</label>
            <select id="prop-cell-type" onchange="updateFreeCell()">
                <option value="text">文本</option>
                <option value="field">数据字段</option>
                <option value="header">表头</option>
            </select>
        </div>
        <div class="property-group">
            <label>内容/字段</label>
            <input type="text" id="prop-cell-value" placeholder="输入文本或字段名" onchange="updateFreeCell()">
        </div>
        <div class="property-group">
            <label>聚合方式（字段时有效）</label>
            <select id="prop-cell-agg" onchange="updateFreeCell()">
                <option value="sum">求和</option>
                <option value="mean">平均</option>
                <option value="count">计数</option>
                <option value="max">最大</option>
                <option value="min">最小</option>
            </select>
        </div>
    `;
}

function updateFreeCell() {
    if (!selectedCell) return;
    const type = document.getElementById('prop-cell-type').value;
    const value = document.getElementById('prop-cell-value').value;
    const agg = document.getElementById('prop-cell-agg').value;
    
    // 更新或添加 cell 配置
    const existing = freeCells.find(c => c.row === selectedCell.row && c.col === selectedCell.col);
    if (existing) {
        existing.type = type;
        existing.value = value;
        existing.field = type === 'field' ? value : null;
        existing.agg = agg;
    } else {
        freeCells.push({
            row: selectedCell.row,
            col: selectedCell.col,
            type: type,
            value: value,
            field: type === 'field' ? value : null,
            agg: agg,
        });
    }
    
    // 更新单元格显示
    const td = document.querySelector(`.free-canvas td[data-row="${selectedCell.row}"][data-col="${selectedCell.col}"]`);
    if (td) {
        td.textContent = value;
        td.classList.toggle('header-cell', type === 'header');
    }
}

function mergeCells() {
    showToast('合并单元格功能在自由报表高级模式中可用', 'info');
}
function splitCell() {
    showToast('拆分单元格功能在自由报表高级模式中可用', 'info');
}
function insertRow() {
    showToast('插入行功能在自由报表高级模式中可用', 'info');
}
function deleteRow() {
    showToast('删除行功能在自由报表高级模式中可用', 'info');
}

// 图表管理
let chartConfigs = [];
let nextChartId = 1;

function addChart() {
    const chartId = nextChartId++;
    const chart = {
        id: chartId,
        chart_type: 'bar',
        x_field: '',
        y_field: '',
        group_field: '',
        agg: 'sum',
        title: `图表 ${chartId}`,
        color_scheme: 'default',
        show_values: true,
        top_n: 0,
    };
    chartConfigs.push(chart);
    renderChartList();
    // 自动滚动到新图表
    setTimeout(() => {
        const el = document.getElementById(`chart-config-${chartId}`);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 50);
}

function removeChart(chartId) {
    chartConfigs = chartConfigs.filter(c => c.id !== chartId);
    renderChartList();
}

function updateChart(chartId, prop, value) {
    const chart = chartConfigs.find(c => c.id === chartId);
    if (chart) {
        chart[prop] = value;
    }
}

function renderChartList() {
    const container = document.getElementById('chart-list');
    if (chartConfigs.length === 0) {
        container.innerHTML = '<p style="color:#999;font-size:13px;">点击"添加图表"创建数据可视化</p>';
        return;
    }
    
    const chartTypes = [
        {value: 'bar', label: '柱状图'},
        {value: 'horizontal_bar', label: '条形图'},
        {value: 'pie', label: '饼图'},
        {value: 'donut', label: '环形图'},
        {value: 'line', label: '折线图'},
        {value: 'area', label: '面积图'},
        {value: 'scatter', label: '散点图'},
    ];
    const aggOptions = [
        {value: 'sum', label: '求和'},
        {value: 'mean', label: '平均'},
        {value: 'count', label: '计数'},
        {value: 'max', label: '最大'},
        {value: 'min', label: '最小'},
    ];
    const colorOptions = [
        {value: 'default', label: '默认'},
        {value: 'pastel', label: '柔和'},
        {value: 'dark', label: '深色'},
        {value: 'bright', label: '明亮'},
    ];
    
    const fieldOptions = reportFields.map(f => `<option value="${f.name}">${f.name}</option>`).join('');
    
    container.innerHTML = chartConfigs.map(chart => `
        <div class="chart-config-item" id="chart-config-${chart.id}">
            <div class="chart-config-header">
                <span class="chart-config-title">${chart.title}</span>
                <div style="display:flex;gap:6px;">
                    <button class="btn btn-sm" onclick="previewSingleChart(${chart.id})">👁 预览</button>
                    <button class="btn btn-sm" style="color:var(--danger);" onclick="removeChart(${chart.id})">🗑️ 删除</button>
                </div>
            </div>
            <div class="chart-config-body">
                <div class="form-group">
                    <label>图表类型</label>
                    <select onchange="updateChart(${chart.id}, 'chart_type', this.value)">
                        ${chartTypes.map(t => `<option value="${t.value}" ${t.value === chart.chart_type ? 'selected' : ''}>${t.label}</option>`).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>标题</label>
                    <input type="text" value="${chart.title}" onchange="updateChart(${chart.id}, 'title', this.value)">
                </div>
                <div class="form-group">
                    <label>X轴/分类字段</label>
                    <select onchange="updateChart(${chart.id}, 'x_field', this.value)">
                        <option value="">请选择</option>
                        ${fieldOptions}
                    </select>
                </div>
                <div class="form-group">
                    <label>Y轴/数值字段</label>
                    <select onchange="updateChart(${chart.id}, 'y_field', this.value)">
                        <option value="">请选择</option>
                        ${fieldOptions}
                    </select>
                </div>
                <div class="form-group">
                    <label>分组字段（可选）</label>
                    <select onchange="updateChart(${chart.id}, 'group_field', this.value)">
                        <option value="">无</option>
                        ${fieldOptions}
                    </select>
                </div>
                <div class="form-group">
                    <label>聚合方式</label>
                    <select onchange="updateChart(${chart.id}, 'agg', this.value)">
                        ${aggOptions.map(o => `<option value="${o.value}" ${o.value === chart.agg ? 'selected' : ''}>${o.label}</option>`).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>配色</label>
                    <select onchange="updateChart(${chart.id}, 'color_scheme', this.value)">
                        ${colorOptions.map(o => `<option value="${o.value}" ${o.value === chart.color_scheme ? 'selected' : ''}>${o.label}</option>`).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>Top N（0=全部）</label>
                    <input type="number" min="0" max="100" value="${chart.top_n}" onchange="updateChart(${chart.id}, 'top_n', parseInt(this.value)||0)">
                </div>
            </div>
            <div class="chart-preview-thumb hidden" id="chart-thumb-${chart.id}"></div>
        </div>
    `).join('');
    
    // 恢复 select 选中状态
    chartConfigs.forEach(chart => {
        const item = document.getElementById(`chart-config-${chart.id}`);
        if (item) {
            item.querySelectorAll('select').forEach(sel => {
                const prop = sel.getAttribute('onchange').match(/updateChart\(\d+,\s*'([^']+)'/)[1];
                sel.value = chart[prop] || '';
            });
        }
    });
}

async function previewSingleChart(chartId) {
    const chart = chartConfigs.find(c => c.id === chartId);
    if (!chart) return;
    if (!chart.x_field || !chart.y_field) {
        showToast('请先配置 X轴 和 Y轴 字段', 'error');
        return;
    }
    
    // 同时在大图预览区显示，复用缩放功能
    const previewArea = document.getElementById('report-preview-area');
    const img = document.getElementById('report-preview-img');
    const titleEl = previewArea.querySelector('h4');
    previewArea.classList.remove('hidden');
    img.src = '';
    img.alt = '正在生成图表...';
    if (titleEl) titleEl.textContent = `图表预览：${chart.title}`;
    resetZoom();
    
    // 滚动到预览区
    previewArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    try {
        const res = await fetch('/api/report/chart/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chart: chart })
        });
        const data = await res.json();
        if (data.success) {
            img.src = data.image;
            img.alt = chart.title;
            showToast('图表生成成功！');
        } else {
            img.alt = '图表生成失败';
            showToast(data.error || '图表生成失败', 'error');
        }
    } catch (e) {
        img.alt = '请求失败';
        showToast('图表请求失败', 'error');
    }
}

// 缩放控制 — 使用宽度百分比，更可靠
const ZOOM_LEVELS = [30, 50, 70, 85, 100, 120, 150, 200, 300, 400, 600, 800, 1000];
let zoomIndex = 4; // 默认 100%

function zoomReport(delta) {
    // delta > 0 放大, delta < 0 缩小
    const step = delta > 0 ? 1 : -1;
    zoomIndex = Math.max(0, Math.min(ZOOM_LEVELS.length - 1, zoomIndex + step));
    applyZoom();
}

function resetZoom() {
    zoomIndex = 4;
    applyZoom();
}

function applyZoom() {
    const img = document.getElementById('report-preview-img');
    const label = document.getElementById('zoom-level');
    const pct = ZOOM_LEVELS[zoomIndex];
    if (img) {
        img.style.width = pct + '%';
    }
    if (label) {
        label.textContent = pct + '%';
    }
}

// 鼠标滚轮缩放
function initZoomWheel() {
    const wrapper = document.getElementById('report-preview-wrapper');
    if (!wrapper) return;
    wrapper.addEventListener('wheel', e => {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            const step = e.deltaY > 0 ? -1 : 1;
            zoomIndex = Math.max(0, Math.min(ZOOM_LEVELS.length - 1, zoomIndex + step));
            applyZoom();
        }
    }, { passive: false });
}

// 构建报表配置
function buildReportConfig() {
    const title = '报表';
    const charts = chartConfigs.map(c => ({
        chart_type: c.chart_type,
        x_field: c.x_field,
        y_field: c.y_field,
        group_field: c.group_field,
        agg: c.agg,
        title: c.title,
        color_scheme: c.color_scheme,
        show_values: c.show_values,
        top_n: c.top_n,
    }));
    
    if (reportMode === 'pivot') {
        return {
            mode: 'pivot',
            title: title,
            pivot: pivotConfig,
            charts: charts,
            styles: {
                header_bg: '#2E86AB',
                header_color: '#FFFFFF',
                grid_color: '#E0E4E8',
                font_size: 11,
                title_font_size: 16,
            }
        };
    } else {
        return {
            mode: 'free',
            title: title,
            cells: freeCells,
            charts: charts,
            row_count: 12,
            col_count: 8,
            styles: {
                header_bg: '#2E86AB',
                header_color: '#FFFFFF',
                grid_color: '#E0E4E8',
                font_size: 11,
                title_font_size: 16,
            }
        };
    }
}

// 预览报表
async function previewReport() {
    const config = buildReportConfig();
    const previewArea = document.getElementById('report-preview-area');
    const img = document.getElementById('report-preview-img');
    const titleEl = previewArea.querySelector('h4');
    
    previewArea.classList.remove('hidden');
    img.src = '';
    img.alt = '正在生成预览...';
    if (titleEl) titleEl.textContent = '预览';
    resetZoom();
    
    try {
        const res = await fetch('/api/report/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config: config })
        });
        const data = await res.json();
        if (data.success) {
            img.src = data.image;
            img.alt = '报表预览';
            showToast('预览生成成功！');
        } else {
            img.alt = '预览失败: ' + (data.error || '未知错误');
            showToast(data.error || '预览失败', 'error');
        }
    } catch (e) {
        img.alt = '请求失败: ' + e.message;
        showToast('预览请求失败', 'error');
    }
}

// 导出报表
async function exportReport(format) {
    const config = buildReportConfig();
    
    try {
        const res = await fetch('/api/report/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config: config, format: format })
        });
        
        if (res.ok && res.headers.get('content-type')?.includes('application/')) {
            // 文件下载
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `报表.${format === 'excel' ? 'xlsx' : 'pdf'}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            showToast('导出成功！');
        } else {
            const data = await res.json();
            showToast(data.error || '导出失败', 'error');
        }
    } catch (e) {
        showToast('导出请求失败', 'error');
    }
}

function renderMarkdown(text) {
    // 简单的 Markdown → HTML 转换器
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // 代码块 ```...```
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

    // 行内代码 `...`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // 标题 ### ## #
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // 引用 >
    html = html.replace(/^&gt; (.*$)/gim, '<blockquote>$1</blockquote>');

    // 无序列表 -
    html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
    // 把连续的 li 包进 ul
    html = html.replace(/(<li>.*<\/li>\n?)+/g, match => '<ul>' + match.replace(/\n/g, '') + '</ul>');

    // 粗体 ** **
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // 斜体 * *
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // 分隔线 ---
    html = html.replace(/^---$/gim, '<hr>');

    // 段落（简单的换行处理）
    const paragraphs = html.split('\n\n');
    html = paragraphs.map(p => {
        p = p.trim();
        if (!p) return '';
        if (p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<pre') || p.startsWith('<blockquote') || p.startsWith('<hr')) {
            return p;
        }
        return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
    }).join('\n');

    return html;
}


// ==================== 运行日志面板 ====================
let logPanelOpen = false;
let logAutoRefreshTimer = null;
let currentLogLevel = 'ALL';

function toggleLogPanel() {
    const panel = document.getElementById('log-panel');
    logPanelOpen = !logPanelOpen;
    if (logPanelOpen) {
        panel.classList.remove('hidden');
        loadLogs();
        startLogAutoRefresh();
    } else {
        panel.classList.add('hidden');
        stopLogAutoRefresh();
    }
}

function startLogAutoRefresh() {
    stopLogAutoRefresh();
    const checkbox = document.getElementById('log-auto-refresh');
    if (checkbox && checkbox.checked) {
        logAutoRefreshTimer = setInterval(() => {
            if (logPanelOpen) loadLogs(false);
        }, 3000);
    }
}

function stopLogAutoRefresh() {
    if (logAutoRefreshTimer) {
        clearInterval(logAutoRefreshTimer);
        logAutoRefreshTimer = null;
    }
}

function toggleLogAutoRefresh() {
    const checkbox = document.getElementById('log-auto-refresh');
    if (checkbox && checkbox.checked) {
        startLogAutoRefresh();
    } else {
        stopLogAutoRefresh();
    }
}

function filterLogs(level) {
    currentLogLevel = level;
    document.querySelectorAll('.log-filter-group .btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.level === level);
    });
    loadLogs();
}

async function loadLogs(showLoading = true) {
    const container = document.getElementById('log-list');
    if (showLoading) container.innerHTML = '<div class="loading">加载日志中...</div>';
    
    try {
        const res = await fetch(`/api/logs?level=${currentLogLevel}&limit=300`);
        const data = await res.json();
        if (data.success) {
            renderLogList(data.logs, data.stats);
        }
    } catch (e) {
        if (showLoading) container.innerHTML = '<div class="loading">加载失败</div>';
    }
}

function renderLogList(logs, stats) {
    const container = document.getElementById('log-list');
    
    // 更新统计数字
    document.getElementById('log-count-all').textContent = stats.total || 0;
    document.getElementById('log-count-debug').textContent = stats.debug || 0;
    document.getElementById('log-count-info').textContent = stats.info || 0;
    document.getElementById('log-count-warning').textContent = stats.warning || 0;
    document.getElementById('log-count-error').textContent = stats.error || 0;
    document.getElementById('log-count-critical').textContent = stats.critical || 0;
    
    if (!logs || logs.length === 0) {
        container.innerHTML = '<div class="log-empty">暂无日志</div>';
        return;
    }
    
    // 终端风格单行渲染
    container.innerHTML = logs.map(log => {
        const time = log.timestamp || '';
        const timePart = time.includes(' ') ? time.split(' ')[1] : time;
        const level = log.level || 'INFO';
        const source = log.source || log.category || 'system';
        const msg = escapeHtml(log.message || '');
        return `<div class="log-entry log-level-${level}"><span class="log-time">[${timePart}]</span> <span class="log-source">[${escapeHtml(source)}]</span> <span class="log-level-${level}">[${level}]</span> ${msg}</div>`;
    }).join('');
    
    // 自动刷新时滚动到顶部（因为后端返回倒序，最新日志在最上方）
    // 只有用户当前已经在顶部附近时才跟随，避免打断用户查看历史日志
    if (logAutoRefreshTimer && container.scrollTop <= 50) {
        container.scrollTop = 0;
    }
}

async function copyAllLogs() {
    const container = document.getElementById('log-list');
    if (!container) return;
    const text = container.innerText || container.textContent || '';
    if (!text.trim() || text.trim() === '加载日志中...') {
        showToast('暂无日志可复制', 'warning');
        return;
    }
    try {
        await navigator.clipboard.writeText(text);
        showToast('日志已复制到剪贴板');
    } catch (e) {
        // 降级方案
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast('日志已复制到剪贴板');
    }
}

async function copyTrainLogs() {
    const container = document.getElementById('train-log-panel');
    if (!container) return;
    const text = container.innerText || container.textContent || '';
    if (!text.trim()) {
        showToast('暂无训练日志可复制', 'warning');
        return;
    }
    try {
        await navigator.clipboard.writeText(text);
        showToast('训练日志已复制到剪贴板');
    } catch (e) {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast('训练日志已复制到剪贴板');
    }
}

async function clearLogs() {
    if (!confirm('确定要清空所有日志吗？')) return;
    try {
        const res = await fetch('/api/logs/clear', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            loadLogs();
            showToast('日志已清空');
        }
    } catch (e) {
        showToast('清空失败', 'error');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


// ==================== 依赖管理面板 ====================
let dependencyPanelOpen = false;

function toggleDependencyPanel() {
    const panel = document.getElementById('dependency-panel');
    dependencyPanelOpen = !dependencyPanelOpen;
    if (dependencyPanelOpen) {
        panel.classList.remove('hidden');
        loadDependencies();
    } else {
        panel.classList.add('hidden');
    }
}

function closeDependencyPanel(event) {
    if (event && event.target !== document.getElementById('dependency-panel')) return;
    document.getElementById('dependency-panel').classList.add('hidden');
    dependencyPanelOpen = false;
}

async function loadDependencies() {
    const container = document.getElementById('dependency-list');
    container.innerHTML = '<div class="loading">检测中...</div>';
    try {
        const res = await fetch('/api/dependencies');
        const data = await res.json();
        if (data.success) {
            renderDependencyList(data.dependencies);
        }
    } catch (e) {
        container.innerHTML = '<div class="loading">加载失败</div>';
    }
}

function renderDependencyList(deps) {
    const container = document.getElementById('dependency-list');
    const missing = deps.filter(d => !d.installed);
    const installed = deps.filter(d => d.installed);
    
    let html = '';
    
    if (missing.length > 0) {
        html += '<div style="margin-bottom:16px;"><strong style="color:#c00;">缺失依赖 (' + missing.length + ')</strong></div>';
        html += '<div style="display:flex;flex-direction:column;gap:10px;">';
        missing.forEach(d => {
            html += `<div class="dependency-item" style="display:flex;align-items:center;justify-content:space-between;padding:10px 12px;background:#fff3f3;border-radius:6px;border:1px solid #ffcdd2;">
                <div>
                    <div style="font-weight:600;">${escapeHtml(d.name)} <code style="background:#eee;padding:2px 6px;border-radius:4px;font-size:12px;">${escapeHtml(d.pip_name)}</code></div>
                    <div style="font-size:12px;color:#666;margin-top:2px;">${escapeHtml(d.description)}</div>
                </div>
                <button class="btn btn-sm btn-primary" onclick="installDependency('${d.key}')">⬇️ 安装</button>
            </div>`;
        });
        html += '</div>';
        html += `<div style="margin-top:12px;text-align:right;"><button class="btn btn-sm" onclick="installAllDependencies()">⬇️ 一键安装全部</button></div>`;
    } else {
        html += '<div style="text-align:center;padding:20px;color:#2e7d32;">✅ 所有可选依赖已安装</div>';
    }
    
    if (installed.length > 0) {
        html += '<div style="margin-top:20px;margin-bottom:8px;"><strong style="color:#2e7d32;">已安装 (' + installed.length + ')</strong></div>';
        html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">';
        installed.forEach(d => {
            html += `<span style="background:#e8f5e9;padding:4px 10px;border-radius:4px;font-size:12px;">✓ ${escapeHtml(d.name)}</span>`;
        });
        html += '</div>';
    }
    
    container.innerHTML = html;
}

function _getTargetDir() {
    const input = document.getElementById('dependency-target-dir');
    return input ? input.value.trim() : '';
}

async function installDependency(packageKey) {
    const targetDir = _getTargetDir();
    showToast(`正在安装 ${packageKey}${targetDir ? ' 到 ' + targetDir : ''}，请稍候...`);
    try {
        const res = await fetch('/api/dependencies/install', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({package: packageKey, target_dir: targetDir || undefined}),
        });
        const data = await res.json();
        if (data.success) {
            showToast(`${packageKey} 安装成功${data.target_dir ? ' (' + data.target_dir + ')' : ''}`);
            loadDependencies();
        } else {
            showToast(`安装失败: ${data.error || data.stderr || '未知错误'}`, 'error');
        }
    } catch (e) {
        showToast('安装请求失败', 'error');
    }
}

async function installAllDependencies() {
    const targetDir = _getTargetDir();
    if (!confirm(`确定要安装所有缺失的依赖吗？${targetDir ? '路径: ' + targetDir : ''}这可能需要几分钟。`)) return;
    showToast('正在批量安装，请稍候...');
    try {
        const res = await fetch('/api/dependencies/install', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({install_all: true, target_dir: targetDir || undefined}),
        });
        const data = await res.json();
        const failed = Object.entries(data.results).filter(([k, v]) => !v.success);
        if (failed.length === 0) {
            showToast('所有依赖安装成功');
        } else {
            showToast(`${failed.length} 个依赖安装失败，请查看详情`, 'error');
        }
        loadDependencies();
    } catch (e) {
        showToast('批量安装失败', 'error');
    }
}


// ==================== 设置面板 ====================
let settingsPanelOpen = false;
let currentSettingsTab = 'network';

function toggleSettingsPanel() {
    const panel = document.getElementById('settings-panel');
    settingsPanelOpen = !settingsPanelOpen;
    if (settingsPanelOpen) {
        panel.classList.remove('hidden');
        loadSettings();
    } else {
        panel.classList.add('hidden');
    }
}

function closeSettingsPanel(event) {
    if (event && event.target !== event.currentTarget) return;
    const panel = document.getElementById('settings-panel');
    panel.classList.add('hidden');
    settingsPanelOpen = false;
}

function switchSettingsTab(tabName) {
    currentSettingsTab = tabName;
    document.querySelectorAll('.settings-tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    document.querySelectorAll('.settings-panel').forEach(p => {
        p.classList.toggle('active', p.id === 'settings-tab-' + tabName);
    });
}

// 加载设置（localStorage + 后端）
async function loadSettings() {
    // 从 localStorage 加载
    const local = JSON.parse(localStorage.getItem('smartchart_settings') || '{}');
    
    // 网络
    if (local.apiBaseUrl) document.getElementById('setting-api-base-url').value = local.apiBaseUrl;
    if (local.httpProxy) document.getElementById('setting-http-proxy').value = local.httpProxy;
    if (local.requestTimeout) document.getElementById('setting-request-timeout').value = local.requestTimeout;
    
    // 侧边栏
    if (local.sidebarVisible !== undefined) document.getElementById('setting-sidebar-visible').checked = local.sidebarVisible;
    if (local.sidebarWidth) {
        document.getElementById('setting-sidebar-width').value = local.sidebarWidth;
        document.getElementById('setting-sidebar-width-val').textContent = local.sidebarWidth;
    }
    if (local.stepVisible) {
        document.querySelectorAll('.setting-step-visible').forEach(cb => {
            cb.checked = local.stepVisible[cb.value] !== false;
        });
    }
    
    // 主题
    if (local.themeMode) {
        document.querySelectorAll('input[name="theme-mode"]').forEach(r => {
            r.checked = r.value === local.themeMode;
        });
    }
    if (local.primaryColor) {
        document.querySelectorAll('.color-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.color === local.primaryColor);
        });
    }
    if (local.fontSize) {
        document.querySelectorAll('input[name="font-size"]').forEach(r => {
            r.checked = r.value === local.fontSize;
        });
    }
    // 背景图片
    if (local.bgImage) {
        document.getElementById('setting-bg-url').value = local.bgImage.url || '';
        updateBgPreview(local.bgImage.url || local.bgImage.dataUrl);
    }
    
    // 系统
    if (local.language) document.getElementById('setting-language').value = local.language;
    if (local.logLevel) document.getElementById('setting-log-level').value = local.logLevel;
    if (local.autoSave !== undefined) document.getElementById('setting-auto-save').checked = local.autoSave;
    if (local.performanceMode !== undefined) document.getElementById('setting-performance-mode').checked = local.performanceMode;
    
    // 侧边栏宽度滑块实时更新
    const widthSlider = document.getElementById('setting-sidebar-width');
    if (widthSlider) {
        widthSlider.addEventListener('input', (e) => {
            document.getElementById('setting-sidebar-width-val').textContent = e.target.value;
        });
    }
    
    // 颜色按钮点击
    document.querySelectorAll('.color-btn').forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        };
    });
    
    // 尝试从后端加载 API Key
    try {
        const res = await fetch('/api/settings/api-key');
        const data = await res.json();
        if (data.success && data.api_key) {
            document.getElementById('setting-api-key').value = data.api_key;
        }
    } catch (e) {
        // 后端未实现时静默失败
    }
}

// 保存设置
async function saveSettings() {
    const settings = {};
    
    // 网络
    settings.apiBaseUrl = document.getElementById('setting-api-base-url').value || 'http://localhost:5000';
    settings.httpProxy = document.getElementById('setting-http-proxy').value || '';
    settings.requestTimeout = parseInt(document.getElementById('setting-request-timeout').value) || 30;
    
    // 侧边栏
    settings.sidebarVisible = document.getElementById('setting-sidebar-visible').checked;
    settings.sidebarWidth = parseInt(document.getElementById('setting-sidebar-width').value) || 220;
    settings.stepVisible = {};
    document.querySelectorAll('.setting-step-visible').forEach(cb => {
        settings.stepVisible[cb.value] = cb.checked;
    });
    
    // 主题
    const themeModeEl = document.querySelector('input[name="theme-mode"]:checked');
    settings.themeMode = themeModeEl ? themeModeEl.value : 'light';
    const colorBtn = document.querySelector('.color-btn.active');
    settings.primaryColor = colorBtn ? colorBtn.dataset.color : '#2E86AB';
    const fontSizeEl = document.querySelector('input[name="font-size"]:checked');
    settings.fontSize = fontSizeEl ? fontSizeEl.value : 'medium';
    // 背景图片
    const bgUrl = document.getElementById('setting-bg-url').value.trim();
    const localData = JSON.parse(localStorage.getItem('smartchart_settings') || '{}');
    settings.bgImage = { url: bgUrl };
    if (localData.bgImage && localData.bgImage.dataUrl && !bgUrl) {
        settings.bgImage.dataUrl = localData.bgImage.dataUrl;
    }
    
    // 系统
    settings.language = document.getElementById('setting-language').value || 'zh-CN';
    settings.logLevel = document.getElementById('setting-log-level').value || 'INFO';
    settings.autoSave = document.getElementById('setting-auto-save').checked;
    settings.performanceMode = document.getElementById('setting-performance-mode').checked;
    
    // 保存到 localStorage
    localStorage.setItem('smartchart_settings', JSON.stringify(settings));
    
    // 应用即时生效的设置
    applySettings(settings);
    
    // 尝试保存到后端
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(settings),
        });
    } catch (e) {
        // 后端未实现时仅保存到 localStorage
    }
    
    showToast('设置已保存');
    closeSettingsPanel();
}

// 应用设置（即时生效）
function applySettings(settings) {
    // 侧边栏宽度
    if (settings.sidebarWidth) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) sidebar.style.width = settings.sidebarWidth + 'px';
    }
    // 侧边栏显隐
    if (settings.sidebarVisible !== undefined) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) sidebar.style.display = settings.sidebarVisible ? '' : 'none';
    }
    // 步骤显隐
    if (settings.stepVisible) {
        Object.entries(settings.stepVisible).forEach(([step, visible]) => {
            const item = document.querySelector(`.step-item[data-step="${step}"]`);
            if (item) item.style.display = visible ? '' : 'none';
        });
    }
    // 主题模式
    if (settings.themeMode) {
        const html = document.documentElement;
        if (settings.themeMode === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            html.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
        } else {
            html.setAttribute('data-theme', settings.themeMode);
        }
    }
    // 主色调
    if (settings.primaryColor) {
        document.documentElement.style.setProperty('--primary', settings.primaryColor);
    }
    // 字体大小
    if (settings.fontSize) {
        const sizes = {small: '13px', medium: '14px', large: '16px'};
        document.body.style.fontSize = sizes[settings.fontSize] || '14px';
    }
    // 性能模式
    if (settings.performanceMode) {
        document.body.classList.add('performance-mode');
    } else {
        document.body.classList.remove('performance-mode');
    }
    // 背景图片
    if (settings.bgImage) {
        const imgUrl = settings.bgImage.url || settings.bgImage.dataUrl || '';
        if (imgUrl) {
            document.body.style.backgroundImage = `url(${imgUrl})`;
            document.body.style.backgroundSize = 'cover';
            document.body.style.backgroundPosition = 'center';
            document.body.style.backgroundAttachment = 'fixed';
        } else {
            document.body.style.backgroundImage = '';
            document.body.style.backgroundSize = '';
            document.body.style.backgroundPosition = '';
            document.body.style.backgroundAttachment = '';
        }
    }
}

function onBgFileSelected(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        const dataUrl = e.target.result;
        document.getElementById('setting-bg-url').value = '';
        updateBgPreview(dataUrl);
        // 保存到 localStorage
        const settings = JSON.parse(localStorage.getItem('smartchart_settings') || '{}');
        settings.bgImage = { dataUrl: dataUrl };
        localStorage.setItem('smartchart_settings', JSON.stringify(settings));
        applySettings(settings);
        showToast('背景图片已应用');
    };
    reader.readAsDataURL(file);
}

function updateBgPreview(url) {
    const preview = document.getElementById('setting-bg-preview');
    if (url) {
        preview.style.backgroundImage = `url(${url})`;
        preview.style.display = 'block';
    } else {
        preview.style.display = 'none';
    }
}

function clearBgImage() {
    document.getElementById('setting-bg-url').value = '';
    document.getElementById('setting-bg-file').value = '';
    updateBgPreview('');
    const settings = JSON.parse(localStorage.getItem('smartchart_settings') || '{}');
    settings.bgImage = { url: '', dataUrl: '' };
    localStorage.setItem('smartchart_settings', JSON.stringify(settings));
    applySettings(settings);
    showToast('背景图片已清除');
}

// 页面加载时应用已保存的设置
document.addEventListener('DOMContentLoaded', () => {
    const saved = JSON.parse(localStorage.getItem('smartchart_settings') || '{}');
    if (Object.keys(saved).length > 0) applySettings(saved);
});

// ==================== 备份与恢复 ====================
async function exportBackup() {
    try {
        const res = await fetch('/api/settings/backup', {method: 'POST'});
        if (!res.ok) throw new Error('导出失败');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `smartchart_backup_${new Date().toISOString().slice(0,10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('备份已导出');
    } catch (e) {
        // 后备方案：导出 localStorage
        const data = {
            settings: JSON.parse(localStorage.getItem('smartchart_settings') || '{}'),
            exported_at: new Date().toISOString(),
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `smartchart_backup_${new Date().toISOString().slice(0,10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('本地配置已导出');
    }
}

async function importBackup(input) {
    const file = input.files[0];
    if (!file) return;
    try {
        const text = await file.text();
        const data = JSON.parse(text);
        if (data.settings) {
            localStorage.setItem('smartchart_settings', JSON.stringify(data.settings));
            applySettings(data.settings);
        }
        showToast('配置已恢复，请刷新页面以完全生效');
        // 尝试后端恢复
        try {
            await fetch('/api/settings/restore', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: text,
            });
        } catch (e) {}
    } catch (e) {
        showToast('备份文件格式错误', 'error');
    }
    input.value = '';
}

function clearAllData() {
    if (!confirm('确定要清空所有本地数据吗？此操作不可逆！')) return;
    localStorage.clear();
    showToast('本地数据已清空，页面即将刷新');
    setTimeout(() => location.reload(), 1500);
}

// ==================== 开发者密钥 ====================
async function generateApiKey() {
    if (!confirm('生成新密钥将撤销旧密钥，确定继续吗？')) return;
    try {
        const res = await fetch('/api/settings/api-key/regenerate', {method: 'POST'});
        const data = await res.json();
        if (data.success && data.api_key) {
            document.getElementById('setting-api-key').value = data.api_key;
            showToast('新密钥已生成');
        } else {
            // 后备：前端生成随机密钥
            const key = 'sk-' + Array.from(crypto.getRandomValues(new Uint8Array(24))).map(b => b.toString(16).padStart(2, '0')).join('');
            document.getElementById('setting-api-key').value = key;
            showToast('新密钥已生成（仅本地）');
        }
    } catch (e) {
        const key = 'sk-' + Array.from(crypto.getRandomValues(new Uint8Array(24))).map(b => b.toString(16).padStart(2, '0')).join('');
        document.getElementById('setting-api-key').value = key;
        showToast('新密钥已生成（仅本地）');
    }
}

async function revokeApiKey() {
    if (!confirm('确定要撤销当前密钥吗？使用此密钥的第三方应用将失效。')) return;
    document.getElementById('setting-api-key').value = '';
    try {
        await fetch('/api/settings/api-key/regenerate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({revoke: true}),
        });
    } catch (e) {}
    showToast('密钥已撤销');
}

function toggleApiKeyVisibility() {
    const input = document.getElementById('setting-api-key');
    input.type = input.type === 'password' ? 'text' : 'password';
}

function copyApiKey() {
    const input = document.getElementById('setting-api-key');
    if (!input.value) { showToast('暂无密钥', 'error'); return; }
    navigator.clipboard.writeText(input.value).then(() => {
        showToast('密钥已复制到剪贴板');
    }).catch(() => {
        input.select();
        document.execCommand('copy');
        showToast('密钥已复制到剪贴板');
    });
}

// ==================== 手动调参面板 ====================
let _tuneSearchSpace = {};
let _tuneModelKey = '';
let _tuneLiveEval = false;
let _tuneEvalTimer = null;
const _TUNE_EVAL_DEBOUNCE_MS = 400;
let _tuneHistory = [];  // { params: {}, score: number, time: number }[]

async function initManualTunePanel() {
    const card = document.getElementById('manual-tune-card');
    if (!card) return;
    
    try {
        const res = await fetch('/api/model/options');
        const data = await res.json();
        if (!data.success || !data.models || data.models.length === 0) {
            card.classList.add('hidden');
            return;
        }
        
        // 过滤出有超参搜索空间的模型
        const modelsWithSpace = data.models.filter(m => {
            const hdata = window._hyperoptData && window._hyperoptData[m.key];
            return hdata && hdata.search_space && Object.keys(hdata.search_space).length > 0;
        });
        
        if (modelsWithSpace.length === 0) {
            card.classList.add('hidden');
            return;
        }
        
        card.classList.remove('hidden');
        
        // 填充下拉框
        const select = document.getElementById('tune-model-select');
        select.innerHTML = '<option value="">-- 选择模型 --</option>' +
            modelsWithSpace.map(m => `<option value="${m.key}">${m.name}</option>`).join('');
        
        // 默认选择第一个
        if (select.options.length > 1) {
            select.selectedIndex = 1;
            onTuneModelChange();
        }
    } catch (e) {
        console.error('initManualTunePanel error', e);
        card.classList.add('hidden');
    }
}

function onTuneModelChange() {
    const select = document.getElementById('tune-model-select');
    const modelKey = select.value;
    if (!modelKey) return;
    
    _tuneModelKey = modelKey;
    const hdata = window._hyperoptData && window._hyperoptData[modelKey];
    if (!hdata || !hdata.search_space) return;
    
    _tuneSearchSpace = hdata.search_space;
    const container = document.getElementById('tune-params-container');
    container.innerHTML = '';
    
    Object.entries(hdata.search_space).forEach(([key, space]) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'form-row';
        wrapper.style.marginBottom = '12px';
        
        let inputHtml = '';
        if (Array.isArray(space)) {
            // 离散值：下拉框，使用 JSON 存储原始类型
            inputHtml = `<select id="tune-param-${key}" onchange="onTuneParamChanged()">` +
                space.map(v => `<option value='${JSON.stringify(v).replace(/'/g, "&#39;")}'>${v}</option>`).join('') +
                '</select>';
        } else if (space && space.type === 'float') {
            // 连续浮点：滑块 + 数字输入
            const low = space.low || 0;
            const high = space.high || 1;
            const step = space.scale === 'log' ? 0.001 : ((high - low) / 100);
            inputHtml = `
                <input type="range" id="tune-param-${key}-range" min="${low}" max="${high}" step="${step}" value="${low}" 
                       oninput="document.getElementById('tune-param-${key}').value=this.value;onTuneParamChanged();" style="flex:2;">
                <input type="number" id="tune-param-${key}" value="${low}" step="${step}" 
                       oninput="document.getElementById('tune-param-${key}-range').value=this.value;onTuneParamChanged();" style="flex:1;min-width:80px;">
            `;
        } else if (space && space.type === 'int') {
            const low = space.low || 0;
            const high = space.high || 100;
            inputHtml = `
                <input type="range" id="tune-param-${key}-range" min="${low}" max="${high}" step="1" value="${low}" 
                       oninput="document.getElementById('tune-param-${key}').value=this.value;onTuneParamChanged();" style="flex:2;">
                <input type="number" id="tune-param-${key}" value="${low}" step="1" 
                       oninput="document.getElementById('tune-param-${key}-range').value=this.value;onTuneParamChanged();" style="flex:1;min-width:80px;">
            `;
        } else {
            inputHtml = `<input type="text" id="tune-param-${key}" value="${JSON.stringify(space)}" onchange="onTuneParamChanged()">`;
        }
        
        wrapper.innerHTML = `
            <div class="form-group" style="flex:1;">
                <label>${key}</label>
                <div style="display:flex;gap:8px;align-items:center;">${inputHtml}</div>
            </div>
        `;
        container.appendChild(wrapper);
    });
    
    // 重置结果区和历史
    document.getElementById('tune-result').classList.add('hidden');
    _tuneHistory = [];
    renderTuneExploreChart();
}

function onLiveEvalToggle() {
    const cb = document.getElementById('tune-live-eval');
    _tuneLiveEval = cb && cb.checked;
    if (_tuneLiveEval && _tuneModelKey) {
        // 开启时立即评估一次当前参数
        onTuneParamChanged();
    }
}

function onTuneParamChanged() {
    if (!_tuneLiveEval || !_tuneModelKey) return;
    // 防抖：连续拖动滑块时只评估最后一次
    if (_tuneEvalTimer) clearTimeout(_tuneEvalTimer);
    _tuneEvalTimer = setTimeout(() => {
        evaluateTuneParams();
    }, _TUNE_EVAL_DEBOUNCE_MS);
}

function resetTuneParams() {
    onTuneModelChange();
    document.getElementById('tune-result').classList.add('hidden');
}

async function evaluateTuneParams() {
    if (!_tuneModelKey) {
        showToast('请先选择模型', 'error');
        return;
    }
    
    const params = {};
    Object.keys(_tuneSearchSpace).forEach(key => {
        const el = document.getElementById(`tune-param-${key}`);
        if (el) {
            let val = el.value;
            // 优先尝试 JSON 解析（处理数组、对象等复杂类型）
            try {
                const parsed = JSON.parse(val);
                if (parsed !== null && typeof parsed !== 'undefined') {
                    val = parsed;
                }
            } catch (e) {
                // 不是 JSON，尝试数字转换
                if (!isNaN(val) && val !== '') {
                    val = val.includes('.') ? parseFloat(val) : parseInt(val, 10);
                }
            }
            params[key] = val;
        }
    });
    
    const btn = document.querySelector('#manual-tune-card .btn-primary');
    btn.disabled = true;
    btn.textContent = '评估中...';
    
    const resultDiv = document.getElementById('tune-result');
    resultDiv.innerHTML = '<div style="color:var(--text-light);">正在评估...</div>';
    resultDiv.classList.remove('hidden');
    
    try {
        const res = await fetch('/api/model/tune', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: _tuneModelKey, params: params })
        });
        const data = await res.json();
        
        if (data.success) {
            const scoreVal = typeof data.score === 'number' ? data.score : 0;
            const scoreStr = scoreVal.toFixed(4);
            resultDiv.innerHTML = `
                <div class="tune-score">${scoreStr}</div>
                <div style="font-size:14px;color:var(--text-light);margin-top:4px;">
                    模型: ${data.model_key || _tuneModelKey}
                </div>
                ${data.cv_scores && data.cv_scores.length ? `<div style="font-size:12px;color:var(--text-light);margin-top:8px;">
                    CV: ${data.cv_scores.map(s => typeof s === 'number' ? s.toFixed(3) : s).join(', ')}
                    (std=${data.std_scores ? Object.values(data.std_scores)[0]?.toFixed(4) : '-'})
                </div>` : ''}
            `;
            // 记录到历史
            _tuneHistory.push({ params: { ...params }, score: scoreVal, time: Date.now() });
            renderTuneExploreChart();
            
            // 渲染诊断信息
            renderTuneDiagnostics(data);
        } else {
            resultDiv.innerHTML = `<div style="color:#dc2626;">评估失败: ${data.error || '未知错误'}</div>`;
        }
    } catch (e) {
        resultDiv.innerHTML = `<div style="color:#dc2626;">请求失败: ${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = '▶️ 评估模型';
    }
}

function renderTuneExploreChart() {
    const chartDom = document.getElementById('tune-explore-chart');
    if (!chartDom) return;
    
    // 清理旧实例
    const existing = echarts.getInstanceByDom(chartDom);
    if (existing) existing.dispose();
    
    if (_tuneHistory.length === 0) {
        chartDom.innerHTML = '<div style="color:var(--text-light);padding:40px;text-align:center;">调整参数并评估，将在此显示参数-性能关系图</div>';
        return;
    }
    
    // 分析参数类型
    const paramKeys = Object.keys(_tuneSearchSpace);
    const numericKeys = paramKeys.filter(k => {
        const space = _tuneSearchSpace[k];
        return (space && (space.type === 'float' || space.type === 'int')) ||
               _tuneHistory.every(h => typeof h.params[k] === 'number');
    });
    const catKeys = paramKeys.filter(k => !numericKeys.includes(k));
    
    const chart = echarts.init(chartDom);
    
    if (numericKeys.length === 1 && catKeys.length === 0) {
        // ===== 单数值参数：散点 + 平滑趋势线 =====
        const pk = numericKeys[0];
        const scatterData = _tuneHistory.map(h => [h.params[pk], h.score]);
        // 按参数值排序后做简单移动平均拟合线
        const sorted = [...scatterData].sort((a, b) => a[0] - b[0]);
        const windowSize = Math.min(3, Math.max(2, Math.floor(sorted.length / 3)));
        const lineData = [];
        for (let i = 0; i < sorted.length; i++) {
            const start = Math.max(0, i - Math.floor(windowSize / 2));
            const end = Math.min(sorted.length, start + windowSize);
            const windowScores = sorted.slice(start, end).map(d => d[1]);
            const avg = windowScores.reduce((a, b) => a + b, 0) / windowScores.length;
            lineData.push([sorted[i][0], avg]);
        }
        
        chart.setOption({
            title: { text: `${pk} → Score`, left: 'center', textStyle: { fontSize: 13 } },
            tooltip: { trigger: 'axis' },
            grid: { left: 50, right: 30, top: 40, bottom: 35 },
            xAxis: { type: 'value', name: pk, nameTextStyle: { fontSize: 11 } },
            yAxis: { type: 'value', name: 'Score', nameTextStyle: { fontSize: 11 } },
            series: [
                { name: '评估点', type: 'scatter', data: scatterData, symbolSize: 14,
                  itemStyle: { color: (p) => {
                      const maxScore = Math.max(..._tuneHistory.map(h => h.score));
                      return p.data[1] >= maxScore * 0.99 ? '#22c55e' : '#3b82f6';
                  }},
                  emphasis: { itemStyle: { borderColor: '#fff', borderWidth: 2 } }
                },
                { name: '趋势', type: 'line', data: lineData, smooth: true, showSymbol: false,
                  lineStyle: { color: '#f59e0b', width: 2, type: 'dashed' }
                }
            ],
            animation: false
        });
    } else if (numericKeys.length >= 2 && catKeys.length === 0) {
        // ===== 双数值参数：散点气泡图 =====
        const pk1 = numericKeys[0];
        const pk2 = numericKeys[1];
        const scatterData = _tuneHistory.map(h => ({
            value: [h.params[pk1], h.params[pk2], h.score],
            itemStyle: { color: h.score >= Math.max(..._tuneHistory.map(x => x.score)) * 0.99 ? '#22c55e' : '#3b82f6' }
        }));
        const sizes = _tuneHistory.map(h => h.score);
        const minSize = Math.min(...sizes), maxSize = Math.max(...sizes);
        const sizeRange = maxSize - minSize || 1;
        
        chart.setOption({
            title: { text: `${pk1} × ${pk2} → Score`, left: 'center', textStyle: { fontSize: 13 } },
            tooltip: { formatter: (p) => `${pk1}: ${p.data.value[0]}<br>${pk2}: ${p.data.value[1]}<br>Score: ${p.data.value[2].toFixed(4)}` },
            grid: { left: 50, right: 30, top: 40, bottom: 35 },
            xAxis: { type: 'value', name: pk1, nameTextStyle: { fontSize: 11 }, scale: true },
            yAxis: { type: 'value', name: pk2, nameTextStyle: { fontSize: 11 }, scale: true },
            visualMap: {
                min: minSize, max: maxSize, dimension: 2,
                inRange: { color: ['#93c5fd', '#3b82f6', '#1e40af'] },
                right: 10, top: 30, itemHeight: 80, textStyle: { fontSize: 10 }
            },
            series: [{
                type: 'scatter',
                data: scatterData,
                symbolSize: (d) => 10 + (d[2] - minSize) / sizeRange * 20,
                emphasis: { itemStyle: { borderColor: '#fff', borderWidth: 2 } }
            }],
            animation: false
        });
    } else if (catKeys.length > 0 && numericKeys.length >= 1) {
        // ===== 含分类参数：分组箱线图（按分类参数分组，显示各组 score） =====
        const catKey = catKeys[0];
        const numKey = numericKeys[0] || catKeys[1];
        const groups = {};
        _tuneHistory.forEach(h => {
            const cat = String(h.params[catKey] ?? 'default');
            if (!groups[cat]) groups[cat] = [];
            groups[cat].push(h);
        });
        const groupNames = Object.keys(groups);
        // 对每个分组做散点
        const series = groupNames.map((gname, idx) => ({
            name: gname,
            type: 'scatter',
            data: groups[gname].map(h => [h.params[numKey] ?? 0, h.score]),
            symbolSize: 12,
            itemStyle: { opacity: 0.8 }
        }));
        
        chart.setOption({
            title: { text: `${catKey} 分组 × ${numKey} → Score`, left: 'center', textStyle: { fontSize: 13 } },
            tooltip: { trigger: 'item' },
            legend: { data: groupNames, top: 22, textStyle: { fontSize: 10 } },
            grid: { left: 50, right: 30, top: 55, bottom: 35 },
            xAxis: { type: 'value', name: numKey, nameTextStyle: { fontSize: 11 } },
            yAxis: { type: 'value', name: 'Score', nameTextStyle: { fontSize: 11 } },
            series: series,
            animation: false
        });
    } else {
        // ===== 其他情况：平行坐标图 =====
        const allKeys = [...numericKeys, ...catKeys];
        // 分类参数映射为数值索引
        const catValueMaps = {};
        catKeys.forEach(ck => {
            const vals = [...new Set(_tuneHistory.map(h => h.params[ck]))];
            catValueMaps[ck] = vals;
        });
        
        const parallelAxis = allKeys.map(k => {
            if (catKeys.includes(k)) {
                const vals = catValueMaps[k];
                return { dim: allKeys.indexOf(k), name: k, type: 'category', data: vals };
            }
            const vals = _tuneHistory.map(h => h.params[k]).filter(v => v !== undefined);
            return { dim: allKeys.indexOf(k), name: k, type: 'value' };
        });
        // score 作为最后一维
        parallelAxis.push({ dim: allKeys.length, name: 'Score', type: 'value' });
        
        const lineData = _tuneHistory.map(h => {
            const row = allKeys.map(k => {
                if (catKeys.includes(k)) return catValueMaps[k].indexOf(h.params[k]);
                return h.params[k] ?? 0;
            });
            row.push(h.score);
            return row;
        });
        
        const scores = _tuneHistory.map(h => h.score);
        const minS = Math.min(...scores), maxS = Math.max(...scores);
        
        chart.setOption({
            title: { text: '参数探索全景', left: 'center', textStyle: { fontSize: 13 } },
            tooltip: { padding: 10, backgroundColor: '#222', borderColor: '#777', textStyle: { color: '#fff' } },
            parallelAxis: parallelAxis,
            parallel: { left: 60, right: 60, top: 50, bottom: 20, parallelAxisDefault: { type: 'value', nameLocation: 'end', nameGap: 20, nameTextStyle: { fontSize: 10 } } },
            series: [{
                type: 'parallel',
                lineStyle: { width: 1, opacity: 0.5 },
                data: lineData,
                // 按 score 着色
                lineStyle: {
                    width: 2,
                    opacity: 0.6,
                    color: (params) => {
                        const score = params.data[params.data.length - 1];
                        const ratio = maxS > minS ? (score - minS) / (maxS - minS) : 0.5;
                        const r = Math.round(59 + ratio * 76);
                        const g = Math.round(130 + ratio * 96);
                        const b = Math.round(246 - ratio * 107);
                        return `rgb(${r},${g},${b})`;
                    }
                }
            }],
            animation: false
        });
    }
    
    setTimeout(() => chart.resize(), 50);
    window.addEventListener('resize', () => chart.resize());
}

function renderTuneDiagnostics(data) {
    const container = document.getElementById('tune-result');
    if (!container || !data.success) return;
    
    let html = container.innerHTML;  // 保留已有的 score 展示
    
    // 每折诊断
    if (data.fold_diagnostics && data.fold_diagnostics.length > 0) {
        html += '<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);"><h5 style="font-size:13px;margin-bottom:8px;">📋 每折诊断</h5><div style="display:flex;gap:8px;flex-wrap:wrap;">';
        data.fold_diagnostics.forEach(fd => {
            const score = typeof fd.score === 'number' ? fd.score.toFixed(4) : fd.score;
            html += `<div style="background:#f1f5f9;padding:6px 10px;border-radius:6px;font-size:12px;">
                <b>Fold ${fd.fold}</b>: ${score}
            </div>`;
        });
        html += '</div></div>';
    }
    
    // 混淆矩阵（分类）
    if (data.diagnostics && data.diagnostics.confusion_matrix) {
        const cm = data.diagnostics.confusion_matrix;
        const labels = data.diagnostics.confusion_matrix_labels || [];
        html += '<div style="margin-top:12px;"><h5 style="font-size:13px;margin-bottom:8px;">🔥 混淆矩阵</h5>';
        html += '<table style="font-size:11px;border-collapse:collapse;margin:0 auto;">';
        html += '<tr><th></th>' + labels.map(l => `<th>${l}</th>`).join('') + '</tr>';
        cm.forEach((row, i) => {
            html += '<tr>' + `<th>${labels[i] !== undefined ? labels[i] : i}</th>` + 
                row.map(v => `<td style="padding:4px 8px;text-align:center;background:${v > 0 ? `rgba(59,130,246,${Math.min(1, v / Math.max(...cm.flat()) * 0.8 + 0.1)})` : '#f8fafc'};color:${v > Math.max(...cm.flat()) * 0.6 ? '#fff' : '#333'};">${v}</td>`).join('') + '</tr>';
        });
        html += '</table></div>';
    }
    
    // 残差（回归）
    if (data.diagnostics && data.diagnostics.residuals) {
        const r = data.diagnostics.residuals;
        html += `<div style="margin-top:12px;"><h5 style="font-size:13px;margin-bottom:8px;">📉 残差统计</h5>
            <div style="display:flex;gap:12px;font-size:12px;">
                <div>Mean: ${r.mean.toFixed(4)}</div>
                <div>Std: ${r.std.toFixed(4)}</div>
                <div>Max|res|: ${r.max_abs.toFixed(4)}</div>
            </div></div>`;
    }
    
    // 特征重要性
    if (data.feature_importance && data.feature_importance.length > 0) {
        html += '<div style="margin-top:12px;"><h5 style="font-size:13px;margin-bottom:8px;">⭐ 特征重要性 (Top 10)</h5>';
        data.feature_importance.slice(0, 10).forEach(fi => {
            const name = fi.feature || fi.Feature || 'Unknown';
            const imp = fi.importance || fi.Importance || 0;
            const pct = Math.min(100, imp * 100).toFixed(1);
            html += `<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;font-size:11px;">
                <span style="width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${name}</span>
                <div style="flex:1;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                    <div style="width:${pct}%;height:100%;background:var(--primary);border-radius:4px;"></div>
                </div>
                <span style="width:40px;text-align:right;font-family:Consolas,monospace;">${pct}%</span>
            </div>`;
        });
        html += '</div>';
    }
    
    container.innerHTML = html;
}

// ==================== 模型排行榜 ====================
function renderModelLeaderboard(leaderboard, taskType, modelScores) {
    const table = document.getElementById('model-leaderboard-table');
    if (!table) return;
    
    // 确定指标列
    let metricCol = 'accuracy_mean';
    let metricName = 'Accuracy';
    if (taskType === 'regression' || taskType === 'REGRESSION') {
        metricCol = 'r2_mean';
        metricName = 'R²';
    } else if (taskType === 'clustering' || taskType === 'CLUSTERING') {
        metricCol = 'silhouette_mean';
        metricName = 'Silhouette';
    }
    
    // 建立 model_key -> scores 映射
    const scoreMap = {};
    if (modelScores && Array.isArray(modelScores)) {
        modelScores.forEach(s => {
            if (s.model_key) scoreMap[s.model_key] = s;
        });
    }
    
    const hasScores = Object.keys(scoreMap).length > 0;
    
    const thead = hasScores
        ? `<tr><th>Rank</th><th>Model</th><th>${metricName}</th><th style="width:120px;">⚡ Speed</th><th style="width:120px;">🎯 Accuracy</th><th>Time</th></tr>`
        : `<tr><th>Rank</th><th>Model</th><th>${metricName}</th><th>Time</th></tr>`;
    
    const tbody = leaderboard.map((m, i) => {
        const score = typeof m[metricCol] === 'number' ? m[metricCol].toFixed(4) : '-';
        const time = m.train_time !== undefined ? m.train_time + 's' : '-';
        const rankClass = i === 0 ? 'gold' : (i === 1 ? 'silver' : (i === 2 ? 'bronze' : ''));
        const medal = i === 0 ? '🥇' : (i === 1 ? '🥈' : (i === 2 ? '🥉' : (i + 1)));
        const modelKey = m.model_key || m.model || '';
        const ms = scoreMap[modelKey];
        
        let extraCols = '';
        if (hasScores && ms) {
            const speedPct = Math.min(100, Math.max(0, ms.speed_score));
            const accPct = Math.min(100, Math.max(0, ms.accuracy_score));
            extraCols = `
                <td><div class="score-bar-wrap" title="Speed: ${ms.speed_score.toFixed(1)}"><div class="score-bar" style="width:${speedPct}%;background:linear-gradient(90deg,#f59e0b,#ef4444);"></div><span class="score-bar-text">${ms.speed_score.toFixed(0)}</span></div></td>
                <td><div class="score-bar-wrap" title="Accuracy: ${ms.accuracy_score.toFixed(1)}"><div class="score-bar" style="width:${accPct}%;background:linear-gradient(90deg,#3b82f6,#22c55e);"></div><span class="score-bar-text">${ms.accuracy_score.toFixed(0)}</span></div></td>
            `;
        } else if (hasScores) {
            extraCols = `<td>-</td><td>-</td>`;
        }
        
        return `<tr><td><span class="live-leaderboard-rank ${rankClass}">${medal}</span></td><td>${escapeHtml(m.model_name || m.model || m.model_key || 'Unknown')}</td><td style="font-family:Consolas,monospace;font-weight:600;">${score}</td>${extraCols}<td>${time}</td></tr>`;
    }).join('');
    
    table.innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
}

let _modelCompTab = 'bar';
let _modelCompLeaderboard = null;
let _modelCompTaskType = null;

function switchModelCompTab(tab) {
    _modelCompTab = tab;
    document.querySelectorAll('#model-comparison-card .chart-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    if (_modelCompLeaderboard) {
        renderModelComparison(_modelCompLeaderboard, _modelCompTaskType);
    }
}

// ==================== 模型对比可视化 ====================
function renderModelComparison(leaderboard, taskType) {
    _modelCompLeaderboard = leaderboard;
    _modelCompTaskType = taskType;
    
    const chartDom = document.getElementById('model-comparison-chart');
    if (!chartDom) return;
    const existing = echarts.getInstanceByDom(chartDom);
    if (existing) existing.dispose();
    
    // 确定主指标
    let metricKey = 'accuracy_mean';
    let metricName = 'Accuracy';
    if (taskType === 'regression' || taskType === 'REGRESSION') {
        metricKey = 'r2_mean';
        metricName = 'R²';
    } else if (taskType === 'clustering' || taskType === 'CLUSTERING') {
        metricKey = 'silhouette_mean';
        metricName = 'Silhouette';
    }
    
    const models = leaderboard.map(m => m.model_name || m.model || m.model_key || 'Unknown');
    
    if (_modelCompTab === 'radar') {
        renderModelRadarChart(chartDom, leaderboard, taskType, models);
    } else if (_modelCompTab === 'box') {
        renderModelBoxChart(chartDom, leaderboard, taskType, models, metricKey);
    } else if (_modelCompTab === 'scatter') {
        renderModelScatterChart(chartDom, leaderboard, taskType, models, metricKey, metricName);
    } else {
        renderModelBarChart(chartDom, leaderboard, models, metricKey, metricName);
    }
}

function renderModelBarChart(chartDom, leaderboard, models, metricKey, metricName) {
    const scores = leaderboard.map(m => {
        const val = m[metricKey];
        return typeof val === 'number' ? val : 0;
    });
    const chart = echarts.init(chartDom);
    chart.setOption({
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        grid: { left: 90, right: 30, top: 20, bottom: 30 },
        xAxis: { type: 'value', name: metricName, nameTextStyle: { fontSize: 11 } },
        yAxis: { type: 'category', data: models.slice().reverse(), axisLabel: { fontSize: 11 }, axisTick: { show: false } },
        series: [{
            type: 'bar',
            data: scores.slice().reverse(),
            barWidth: 16,
            itemStyle: {
                borderRadius: [0, 4, 4, 0],
                color: (p) => {
                    const maxScore = Math.max(...scores);
                    const ratio = maxScore > 0 ? p.data / maxScore : 0;
                    const r = Math.round(147 + (1 - ratio) * 108);
                    const g = Math.round(197 + (1 - ratio) * 58);
                    const b = Math.round(253 - ratio * 46);
                    return `rgb(${r},${g},${b})`;
                }
            },
            label: { show: true, position: 'right', fontSize: 11, formatter: (p) => p.data.toFixed(3) }
        }],
        animation: true
    });
    setTimeout(() => chart.resize(), 50);
    window.addEventListener('resize', () => chart.resize());
}

function renderModelRadarChart(chartDom, leaderboard, taskType, models) {
    let dims = [];
    if (taskType === 'classification' || taskType === 'CLASSIFICATION') {
        dims = [
            {key: 'accuracy_mean', name: 'Accuracy', max: 1.0},
            {key: 'precision_mean', name: 'Precision', max: 1.0},
            {key: 'recall_mean', name: 'Recall', max: 1.0},
            {key: 'f1_mean', name: 'F1', max: 1.0},
            {key: 'auc_mean', name: 'AUC', max: 1.0}
        ];
    } else if (taskType === 'regression' || taskType === 'REGRESSION') {
        dims = [
            {key: 'r2_mean', name: 'R²', max: 1.0},
            {key: 'rmse_mean', name: 'RMSE↓', max: null},
            {key: 'mae_mean', name: 'MAE↓', max: null}
        ];
    } else {
        dims = [
            {key: 'silhouette_mean', name: 'Silhouette', max: 1.0},
            {key: 'calinski_mean', name: 'Calinski', max: null},
            {key: 'davies_bouldin_mean', name: 'DB↓', max: null}
        ];
    }
    
    // 过滤实际存在的维度，并计算 max
    const availableDims = [];
    dims.forEach(d => {
        const hasVal = leaderboard.some(m => typeof m[d.key] === 'number');
        if (hasVal) {
            let maxVal = d.max;
            if (maxVal === null) {
                const vals = leaderboard.map(m => m[d.key]).filter(v => typeof v === 'number');
                maxVal = vals.length > 0 ? Math.max(...vals) * 1.1 : 1;
                if (maxVal <= 0) maxVal = 1;
            }
            availableDims.push({...d, max: maxVal});
        }
    });
    
    if (availableDims.length === 0) {
        chartDom.innerHTML = '<div style="text-align:center;padding:80px 0;color:var(--text-light);">No multi-dimensional metrics available</div>';
        return;
    }
    
    // 归一化每个维度
    const indicator = availableDims.map(d => ({name: d.name, max: d.max}));
    const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272'];
    const seriesData = leaderboard.slice(0, 6).map((m, i) => {
        const vals = availableDims.map(d => {
            const v = m[d.key];
            if (typeof v !== 'number') return 0;
            if (d.key.includes('rmse') || d.key.includes('mae') || d.key.includes('davies')) {
                // 越小越好，反转归一化
                return Math.max(0, 1 - v / d.max);
            }
            return Math.min(1, Math.max(0, v / d.max));
        });
        return {
            value: vals,
            name: models[leaderboard.indexOf(m)],
            lineStyle: { color: colors[i % colors.length] },
            areaStyle: { color: colors[i % colors.length], opacity: 0.1 }
        };
    });
    
    const chart = echarts.init(chartDom);
    chart.setOption({
        tooltip: { trigger: 'item' },
        legend: { data: seriesData.map(d => d.name), top: 0, textStyle: { fontSize: 11 } },
        radar: {
            indicator: indicator,
            radius: '65%',
            center: ['50%', '55%'],
            axisName: { fontSize: 11 }
        },
        series: [{
            type: 'radar',
            data: seriesData,
            symbolSize: 4
        }],
        animation: true
    });
    setTimeout(() => chart.resize(), 50);
    window.addEventListener('resize', () => chart.resize());
}

function renderModelBoxChart(chartDom, leaderboard, taskType, models, metricKey) {
    const foldKey = metricKey.replace('_mean', '_fold_scores');
    const hasFold = leaderboard.some(m => Array.isArray(m[foldKey]) && m[foldKey].length > 0);
    
    if (!hasFold) {
        chartDom.innerHTML = '<div style="text-align:center;padding:80px 0;color:var(--text-light);">No per-fold scores available</div>';
        return;
    }
    
    // 构建箱线图数据（过滤无数据的模型）
    const validModels = [];
    const boxData = [];
    const outlierData = [];
    models.forEach((name, idx) => {
        const arr = leaderboard[idx][foldKey];
        if (!Array.isArray(arr) || arr.length === 0) return;
        validModels.push(name);
        const sorted = [...arr].sort((a, b) => a - b);
        const q1 = sorted[Math.floor(sorted.length * 0.25)];
        const q3 = sorted[Math.ceil(sorted.length * 0.75) - 1];
        const median = sorted[Math.floor(sorted.length * 0.5)];
        const min = sorted[0];
        const max = sorted[sorted.length - 1];
        boxData.push([min, q1, median, q3, max]);
        // 异常值
        const iqr = q3 - q1;
        const lower = q1 - 1.5 * iqr;
        const upper = q3 + 1.5 * iqr;
        arr.forEach(v => {
            if (v < lower || v > upper) {
                outlierData.push([name, v]);
            }
        });
    });
    
    if (validModels.length === 0) {
        chartDom.innerHTML = '<div style="text-align:center;padding:80px 0;color:var(--text-light);">No per-fold scores available</div>';
        return;
    }
    
    const chart = echarts.init(chartDom);
    chart.setOption({
        tooltip: { trigger: 'item' },
        grid: { left: 90, right: 30, top: 20, bottom: 30 },
        xAxis: { type: 'value', name: 'Score', nameTextStyle: { fontSize: 11 } },
        yAxis: { type: 'category', data: validModels, axisLabel: { fontSize: 11 }, axisTick: { show: false } },
        series: [
            {
                type: 'boxplot',
                data: boxData,
                itemStyle: { color: '#91cc75', borderColor: '#5470c6' }
            },
            {
                type: 'scatter',
                data: outlierData,
                symbolSize: 8,
                itemStyle: { color: '#ee6666' }
            }
        ],
        animation: true
    });
    setTimeout(() => chart.resize(), 50);
    window.addEventListener('resize', () => chart.resize());
}

function renderModelScatterChart(chartDom, leaderboard, taskType, models, metricKey, metricName) {
    const data = leaderboard.map((m, i) => {
        const score = typeof m[metricKey] === 'number' ? m[metricKey] : 0;
        const time = typeof m.train_time === 'number' ? m.train_time : (parseFloat(m.train_time) || 0);
        return [time, score, models[i]];
    }).filter(d => d[1] !== 0);
    
    if (data.length === 0) {
        chartDom.innerHTML = '<div style="text-align:center;padding:80px 0;color:var(--text-light);">No data available</div>';
        return;
    }
    
    // 计算 Pareto 前沿：time 越小越好，score 越大越好
    // 按 time 升序，维护当前最大 score
    const sorted = [...data].sort((a, b) => a[0] - b[0]);
    const pareto = [];
    let maxScore = -Infinity;
    for (const pt of sorted) {
        if (pt[1] >= maxScore) {
            pareto.push(pt);
            maxScore = pt[1];
        }
    }
    // 确保线从 x=0 附近开始（取第一个点的 time 和 score）
    const paretoLine = pareto.map(p => [p[0], p[1]]);
    
    const chart = echarts.init(chartDom);
    chart.setOption({
        tooltip: {
            formatter: (p) => {
                if (p.seriesName === 'Pareto') return `Pareto frontier<br/>Time: ${p.data[0].toFixed(2)}s<br/>${metricName}: ${p.data[1].toFixed(4)}`;
                return `<b>${p.data[2]}</b><br/>Time: ${p.data[0].toFixed(2)}s<br/>${metricName}: ${p.data[1].toFixed(4)}`;
            }
        },
        legend: { data: ['Models', 'Pareto frontier'], top: 0, textStyle: { fontSize: 11 } },
        grid: { left: 60, right: 30, top: 30, bottom: 40 },
        xAxis: { type: 'value', name: 'Train Time (s)', nameTextStyle: { fontSize: 11 } },
        yAxis: { type: 'value', name: metricName, nameTextStyle: { fontSize: 11 } },
        series: [
            {
                name: 'Models',
                type: 'scatter',
                data: data,
                symbolSize: (d) => Math.max(10, Math.min(30, d[1] * 20)),
                itemStyle: {
                    color: (p) => {
                        const ratio = p.data[1] / Math.max(...data.map(d => d[1]));
                        const r = Math.round(147 + (1 - ratio) * 108);
                        const g = Math.round(197 + (1 - ratio) * 58);
                        const b = Math.round(253 - ratio * 46);
                        return `rgb(${r},${g},${b})`;
                    }
                },
                label: { show: true, position: 'top', fontSize: 10, formatter: (p) => p.data[2], color: 'var(--text)' }
            },
            {
                name: 'Pareto frontier',
                type: 'line',
                data: paretoLine,
                smooth: false,
                lineStyle: { color: '#ef4444', width: 2, type: 'dashed' },
                itemStyle: { color: '#ef4444' },
                symbol: 'none',
                silent: true
            }
        ],
        animation: true
    });
    setTimeout(() => chart.resize(), 50);
    window.addEventListener('resize', () => chart.resize());
}


async function loadSHAPExplanation(modelKey) {
    const card = document.getElementById('shap-card');
    const container = document.getElementById('shap-content');
    if (!card || !container) return;
    container.innerHTML = '<div class="hint">Loading SHAP explanation...</div>';
    card.classList.remove('hidden');
    try {
        const res = await fetch('/api/model/explain/shap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: modelKey })
        });
        const data = await res.json();
        if (data.success && data.shap) {
            renderSHAP(data.shap);
        } else {
            container.innerHTML = '<div class="hint">SHAP not available: ' + (data.error || '') + '</div>';
        }
    } catch (e) {
        container.innerHTML = '<div class="hint">SHAP error: ' + e.message + '</div>';
    }
}

function renderSHAP(shapData) {
    const chartDom = document.getElementById('shap-chart');
    const container = document.getElementById('shap-content');
    if (!container) return;
    // Render bar chart
    if (chartDom && shapData.feature_importance) {
        const existing = echarts.getInstanceByDom(chartDom);
        if (existing) existing.dispose();
        const sorted = [...shapData.feature_importance].sort((a, b) => a.importance - b.importance).slice(-15);
        const chart = echarts.init(chartDom);
        chart.setOption({
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            grid: { left: 100, right: 30, top: 10, bottom: 20 },
            xAxis: { type: 'value', name: '|SHAP|', nameTextStyle: { fontSize: 11 } },
            yAxis: { type: 'category', data: sorted.map(d => d.feature), axisLabel: { fontSize: 10 } },
            series: [{ type: 'bar', data: sorted.map(d => d.importance), barWidth: 14, itemStyle: { color: '#8e44ad', borderRadius: [0, 4, 4, 0] } }]
        });
        setTimeout(() => chart.resize(), 50);
    }
    // Render table content
    let html = '<div class="config-display">';
    html += '<div class="config-item"><div class="label">Background</div><div class="value">' + shapData.n_background + '</div></div>';
    html += '<div class="config-item"><div class="label">Explained</div><div class="value">' + shapData.n_explained + '</div></div>';
    html += '</div>';
    if (shapData.instance_explanations && shapData.instance_explanations.length > 0) {
        html += '<h4 style="margin:12px 0 6px;font-size:13px;">Sample Explanations</h4>';
        shapData.instance_explanations.forEach(inst => {
            html += '<p style="font-size:12px;margin:4px 0;"><strong>Instance #' + inst.index + '</strong></p>';
            html += '<table class="data-table" style="font-size:11px;"><thead><tr><th>Feature</th><th>Value</th><th>SHAP</th></tr></thead><tbody>';
            inst.top_features.forEach(tf => {
                const color = tf.shap > 0 ? 'color:#27ae60;' : (tf.shap < 0 ? 'color:#e74c3c;' : '');
                html += '<tr><td>' + tf.feature + '</td><td>' + tf.value + '</td><td style="' + color + '">' + tf.shap.toFixed(4) + '</td></tr>';
            });
            html += '</tbody></table>';
        });
    }
    container.innerHTML = html;
}

async function runIncrementalLearning() {
    const resultDiv = document.getElementById('incremental-result');
    if (resultDiv) resultDiv.textContent = 'Updating...';
    try {
        const res = await fetch('/api/model/incremental', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: selectedOverrideModel || currentBestModelKey })
        });
        const data = await res.json();
        if (resultDiv) resultDiv.textContent = data.success ? 'Model updated successfully.' : ('Failed: ' + (data.error || ''));
        if (data.success) showToast('Model updated', 'success');
        else showToast(data.error || 'Update failed', 'error');
    } catch (e) {
        if (resultDiv) resultDiv.textContent = 'Error: ' + e.message;
        showToast('Incremental learning error: ' + e.message, 'error');
    }
}

async function generateReport(format) {
    try {
        const res = await fetch('/api/model/report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ format: format })
        });
        const data = await res.json();
        if (data.success && data.html) {
            const w = window.open('', '_blank');
            w.document.write(data.html);
            w.document.close();
            showToast('Report opened in new tab', 'success');
        } else if (data.success && data.path) {
            showToast('Report saved: ' + data.path, 'success');
        } else {
            showToast(data.error || 'Report failed', 'error');
        }
    } catch (e) {
        showToast('Report error: ' + e.message, 'error');
    }
}

async function deployModel() {
    const modelKey = selectedOverrideModel || currentBestModelKey;
    if (!modelKey) {
        showToast('Please select a model first', 'error');
        return;
    }
    try {
        const res = await fetch('/api/model/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: modelKey })
        });
        const data = await res.json();
        if (data.success && data.files) {
            showToast('Deploy package ready: ' + Object.keys(data.files).join(', '), 'success');
        } else {
            showToast(data.error || 'Deploy failed', 'error');
        }
    } catch (e) {
        showToast('Deploy error: ' + e.message, 'error');
    }
}

async function runRobustnessTest() {
    const modelKey = selectedOverrideModel || currentBestModelKey;
    if (!modelKey) {
        showToast('Please select a model first', 'error');
        return;
    }
    const container = document.getElementById('robustness-content');
    if (container) container.innerHTML = 'Running tests...';
    try {
        const res = await fetch('/api/model/robustness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: modelKey })
        });
        const data = await res.json();
        if (data.success && data.robustness) {
            const r = data.robustness;
            let html = '<div class="config-display">';
            html += '<div class="config-item"><div class="label">Baseline</div><div class="value">' + r.baseline + '</div></div>';
            html += '<div class="config-item"><div class="label">Robustness Score</div><div class="value">' + r.robustness_score + '/100 (' + r.summary + ')</div></div>';
            html += '</div>';
            html += '<table class="data-table" style="font-size:12px;"><thead><tr><th>Test</th><th>Score</th><th>Drop</th></tr></thead><tbody>';
            r.tests.forEach(t => {
                html += '<tr><td>' + t.name + '</td><td>' + t.score + '</td><td>' + t.drop + '</td></tr>';
            });
            html += '</tbody></table>';
            if (container) container.innerHTML = html;
        } else {
            if (container) container.innerHTML = 'Failed: ' + (data.error || '');
        }
    } catch (e) {
        if (container) container.innerHTML = 'Error: ' + e.message;
    }
}

async function saveModelSnapshot() {
    const modelKey = selectedOverrideModel || currentBestModelKey;
    if (!modelKey) {
        showToast('Please select a model first', 'error');
        return;
    }
    try {
        const res = await fetch('/api/model/snapshot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_key: modelKey })
        });
        const data = await res.json();
        if (data.success) showToast('Snapshot saved #' + data.snapshot_id, 'success');
        else showToast(data.error || 'Save failed', 'error');
    } catch (e) {
        showToast('Snapshot error: ' + e.message, 'error');
    }
}

function renderPermutationImportance(data) {
    const chartDom = document.getElementById('permutation-importance-chart');
    if (!chartDom) return;
    const existing = echarts.getInstanceByDom(chartDom);
    if (existing) existing.dispose();
    
    // 取 Top 20，按重要性升序（图表从上到下）
    const sorted = [...data].sort((a, b) => (a.importance || 0) - (b.importance || 0)).slice(-20);
    const features = sorted.map(d => d.feature || d.Feature || 'Unknown');
    const importances = sorted.map(d => typeof d.importance === 'number' ? d.importance : 0);
    const stds = sorted.map(d => typeof d.std === 'number' ? d.std : 0);
    
    const chart = echarts.init(chartDom);
    chart.setOption({
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        grid: { left: 100, right: 30, top: 10, bottom: 20 },
        xAxis: { type: 'value', name: 'Importance', nameTextStyle: { fontSize: 11 } },
        yAxis: { type: 'category', data: features, axisLabel: { fontSize: 10 }, axisTick: { show: false } },
        series: [{
            type: 'bar',
            data: importances.map((v, i) => ({ value: v, itemStyle: { color: v >= 0 ? '#3b82f6' : '#ef4444' } })),
            barWidth: 14,
            itemStyle: { borderRadius: [0, 4, 4, 0] },
            label: { show: true, position: 'right', fontSize: 10, formatter: (p) => p.data.value.toFixed(3) },
            errorBars: stds.map((s, i) => ({ xAxis: s }))
        }],
        animation: true
    });
    setTimeout(() => chart.resize(), 50);
    window.addEventListener('resize', () => chart.resize());
}
