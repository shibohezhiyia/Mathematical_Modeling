/**
 * Mathematical Modeling - 前端主逻辑
 */

// ==================== 全局状态 ====================
let currentStep = 1;
let uploadedData = null;
let trainTimer = null;
let trainEventSinceId = -1;
let selectedOverrideModel = null;
let llmTimer = null;
let selectedAnalysisType = null;
let llmImageAttachments = [];
let researchImageAttachments = [];
let interactiveVizSchema = null;
let interactiveVizResult = null;
let interactiveVizChart = null;
let interactiveVizRefreshTimer = null;
let interactiveVizPlaybackTimer = null;
let interactiveVizRequestId = 0;
let transformPreviewChart = null;
let transformPreviewResult = null;
let transformPreviewPresets = [];
let transformPreviewView = { orientation: 'horizontal', sort: 'desc', topN: 15, labels: true };
let transformEditingStepIndex = null;

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initPredictUpload();
    initModeHint();
    initDragDrop();
    loadDatasets();  // 加载已有数据集列表
    loadTransformCapabilities();
    onResearchSemanticProviderChange();
    initResearchImageUpload();
});

window.addEventListener('resize', () => {
    if (transformPreviewChart) transformPreviewChart.resize();
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
let transformCapabilities = [];
let transformRecommendations = [];
let transformValidationSuggestion = null;
let mathematicalDataCompilation = null;

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
            html += '<h4 style="margin:0 0 8px;color:#2c3e50;">' + escapeHtml(r.model_class || r.model || '建模任务') + ' <span style="font-size:12px;color:#27ae60;">(' + r.confidence + '% 匹配)</span></h4>';
            html += '<h5 style="margin:10px 0 4px;font-size:13px;">关键公式</h5><ul style="font-size:12px;margin:0;padding-left:16px;">';
            (r.formulas || []).forEach(f => { html += '<li>' + escapeHtml(f) + '</li>'; });
            html += '</ul>';
            html += '<h5 style="margin:10px 0 4px;font-size:13px;">建模步骤</h5><ol style="font-size:12px;margin:0;padding-left:16px;">';
            (r.approach || r.steps || []).forEach(a => { html += '<li>' + escapeHtml(a) + '</li>'; });
            html += '</ol>';
            html += '<h5 style="margin:10px 0 4px;font-size:13px;">关键变量</h5><p style="font-size:12px;margin:0;">' + r.key_features.join('、') + '</p>';
            html += '<h5 style="margin:10px 0 4px;font-size:13px;">Python 代码框架</h5><pre style="background:#fff;padding:8px;border-radius:4px;font-size:11px;overflow:auto;border:1px solid #e5e7eb;">' + escapeHtml(r.code_template || r.code_framework || '') + '</pre>';
            html += '</div>';
            if (contentDiv) contentDiv.innerHTML = html;
        } else {
            if (contentDiv) contentDiv.innerHTML = '<div class="hint">分析失败: ' + (data.error || '') + '</div>';
        }
    } catch (e) {
        if (contentDiv) contentDiv.innerHTML = '<div class="hint">错误: ' + e.message + '</div>';
    }
}

function initResearchImageUpload() {
    const input = document.getElementById('research-image-input');
    if (!input || input.dataset.bound === '1') return;
    input.dataset.bound = '1';
    input.addEventListener('change', async event => {
        const files = Array.from(event.target.files || []);
        event.target.value = '';
        if (!files.length) return;
        const remaining = LLM_IMAGE_MAX_COUNT - researchImageAttachments.length;
        if (remaining <= 0) {
            showToast(`最多上传 ${LLM_IMAGE_MAX_COUNT} 张题图`, 'error');
            return;
        }
        for (const file of files.slice(0, remaining)) {
            if (!/^image\/(png|jpe?g|webp|gif)$/i.test(file.type)) {
                showToast(`${file.name} 不是支持的图片格式`, 'error');
                continue;
            }
            if (file.size <= 0 || file.size > LLM_IMAGE_MAX_BYTES) {
                showToast(`${file.name} 超过 6 MB 限制`, 'error');
                continue;
            }
            const total = researchImageAttachments.reduce((sum, image) => sum + image.size, 0);
            if (total + file.size > LLM_IMAGE_TOTAL_MAX_BYTES) {
                showToast('题图总大小不能超过 20 MB', 'error');
                break;
            }
            try {
                const dataUrl = await readFileAsDataUrl(file);
                researchImageAttachments.push({ name: file.name, mime_type: file.type.toLowerCase(), size: file.size, data_url: dataUrl });
            } catch (error) {
                showToast(`读取 ${file.name} 失败`, 'error');
            }
        }
        renderResearchImagePreview();
    });
    renderResearchImagePreview();
}

function removeResearchImage(index) {
    researchImageAttachments.splice(index, 1);
    renderResearchImagePreview();
}

function renderResearchImagePreview() {
    const container = document.getElementById('research-image-preview');
    const status = document.getElementById('research-image-status');
    if (!container || !status) return;
    container.innerHTML = '';
    if (!researchImageAttachments.length) {
        status.textContent = '未选择图片';
        return;
    }
    status.textContent = `已选择 ${researchImageAttachments.length}/${LLM_IMAGE_MAX_COUNT} 张题图`;
    researchImageAttachments.forEach((image, index) => {
        const card = document.createElement('div');
        card.className = 'llm-image-card';
        const preview = document.createElement('img');
        preview.className = 'llm-image-thumb';
        preview.src = image.data_url;
        preview.alt = image.name;
        const meta = document.createElement('div');
        meta.className = 'llm-image-meta';
        const name = document.createElement('span');
        name.textContent = image.name;
        const size = document.createElement('small');
        size.textContent = `${(image.size / 1024).toFixed(0)} KB`;
        meta.append(name, size);
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'llm-image-remove';
        remove.title = '移除题图';
        remove.textContent = '×';
        remove.addEventListener('click', () => removeResearchImage(index));
        card.append(preview, meta, remove);
        container.appendChild(card);
    });
}

function onResearchSemanticProviderChange() {
    const provider = document.getElementById('research-semantic-provider').value;
    const baseUrl = document.getElementById('research-semantic-base-url');
    const modelName = document.getElementById('research-semantic-model-name');
    const keyGroup = document.getElementById('research-semantic-key-group');
    if (provider === 'ollama') {
        baseUrl.value = 'http://localhost:11434';
        modelName.value = 'qwen2.5:3b';
        keyGroup.classList.add('hidden');
    } else if (provider === 'local_openai') {
        baseUrl.value = 'http://localhost:1234/v1';
        modelName.value = 'local-model';
        keyGroup.classList.add('hidden');
    } else if (provider === 'deepseek') {
        baseUrl.value = 'https://api.deepseek.com';
        modelName.value = 'deepseek-v4-pro';
        keyGroup.classList.remove('hidden');
    } else {
        baseUrl.value = 'https://api.openai.com/v1';
        modelName.value = 'gpt-4o-mini';
        keyGroup.classList.remove('hidden');
    }
    const status = document.getElementById('research-semantic-test-status');
    if (status) status.textContent = '';
    populateModelPresetSelect('research-semantic-model-preset', provider, modelName.value);
}

const BUILTIN_MODEL_PRESETS = {
    ollama: ['qwen2.5:3b', 'llama3', 'deepseek-r1:7b'],
    local_openai: ['local-model'],
    openai_compatible: ['gpt-4o', 'gpt-4o-mini'],
    openai: ['gpt-4o', 'gpt-4o-mini'],
    deepseek: ['deepseek-v4-pro', 'deepseek-v4-flash', 'deepseek-v4-flash-vision-exp'],
};

function populateModelPresetSelect(selectId, provider, currentValue = '') {
    const select = document.getElementById(selectId);
    if (!select) return;
    const remoteConfig = window.llmProviders?.[provider] || {};
    const options = Array.from(new Set([...(remoteConfig.model_options || []), ...(BUILTIN_MODEL_PRESETS[provider] || [])].filter(Boolean)));
    select.innerHTML = '<option value="">选择常用模型…</option>' + options.map(model => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join('');
    select.value = options.includes(currentValue) ? currentValue : '';
}

function mergeRemoteModelPresetOptions(selectId, models, currentValue = '') {
    const select = document.getElementById(selectId);
    if (!select || !Array.isArray(models) || !models.length) return;
    const existing = Array.from(select.options).map(option => option.value).filter(Boolean);
    const options = Array.from(new Set([...existing, ...models.map(String).filter(Boolean)]));
    select.innerHTML = '<option value="">选择常用模型…</option>' + options.map(model => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join('');
    select.value = options.includes(currentValue) ? currentValue : '';
}

function useResearchSemanticModelPreset(value) {
    if (!value) return;
    const input = document.getElementById('research-semantic-model-name');
    if (input) input.value = value;
}

async function requestLLMConnectionTest(config, button, status) {
    if (!config.base_url || !config.model_name) {
        showToast('请填写服务地址和模型名称', 'error');
        return false;
    }
    if (config.provider === 'deepseek' && !config.api_key) {
        showToast('请输入 DeepSeek API Key', 'error');
        return false;
    }
    if (button) button.disabled = true;
    if (status) {
        status.textContent = '正在验证…';
        status.style.color = 'var(--text-light)';
    }
    try {
        const response = await fetch('/api/llm/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || '连接失败');
        if (status) {
            status.textContent = data.message;
            status.style.color = data.model_available ? '#16845b' : '#a26805';
        }
        const presetId = button?.id === 'research-semantic-test-btn'
            ? 'research-semantic-model-preset'
            : 'llm-model-preset';
        mergeRemoteModelPresetOptions(presetId, data.available_models, config.model_name);
        showToast(data.message, data.model_available ? 'success' : 'warning');
        return true;
    } catch (error) {
        if (status) {
            status.textContent = error.message;
            status.style.color = '#c0392b';
        }
        showToast('API 验证失败: ' + error.message, 'error');
        return false;
    } finally {
        if (button) button.disabled = false;
    }
}

function testResearchSemanticConnection() {
    return requestLLMConnectionTest({
        provider: document.getElementById('research-semantic-provider').value,
        base_url: document.getElementById('research-semantic-base-url').value.trim(),
        model_name: document.getElementById('research-semantic-model-name').value.trim(),
        api_key: document.getElementById('research-semantic-api-key').value,
    }, document.getElementById('research-semantic-test-btn'), document.getElementById('research-semantic-test-status'));
}

async function runResearch() {
    const description = document.getElementById('problem-description').value.trim();
    if (!description) {
        showToast('请先粘贴完整题目', 'error');
        return;
    }
    const button = document.getElementById('research-run-btn');
    const progress = document.getElementById('research-progress');
    const resultBox = document.getElementById('research-result');
    const hasDataset = uploadedFiles.length > 0 || Boolean(uploadedData);
    const messages = [
        hasDataset ? '正在读取全部数据表并建立字段画像…' : '正在从题面抽取实体、参数、单位与数学关系…',
        hasDataset ? '正在推断主表、明细表和跨表关联键…' : '正在构建题目无关的数学中间表示…',
        hasDataset ? '正在计算跨数据集变量交互，控制联表膨胀…' : '正在组合动力学、几何、事件和优化算子…',
        '正在检查变量、单位、假设、初边值和约束…',
        '正在安全编译可执行部分并保留未决条件…',
        '正在执行泄漏、置乱、稳定性和敏感性反证…',
        '正在编译论证图、结论等级和数学证据包…'
    ];
    let messageIndex = 0;
    button.disabled = true;
    progress.classList.remove('hidden');
    resultBox.classList.add('hidden');
    progress.innerHTML = `<span class="research-spinner"></span><span>${messages[0]}</span>`;
    const timer = setInterval(() => {
        messageIndex = Math.min(messageIndex + 1, messages.length - 1);
        progress.innerHTML = `<span class="research-spinner"></span><span>${messages[messageIndex]}</span>`;
    }, 3500);
    try {
        const target = document.getElementById('research-target').value.trim();
        const semanticEnabled = document.getElementById('research-semantic-model').checked;
        const response = await fetch('/api/research/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                description,
                target: target || null,
                run_modeling: document.getElementById('research-run-model').checked,
                feedback_optimization: document.getElementById('research-feedback-optimize').checked,
                credibility_audit: document.getElementById('research-credibility-audit').checked,
                semantic_model_compiler: semanticEnabled,
                semantic_provider: document.getElementById('research-semantic-provider').value,
                semantic_base_url: document.getElementById('research-semantic-base-url').value.trim(),
                semantic_model_name: document.getElementById('research-semantic-model-name').value.trim(),
                semantic_api_key: semanticEnabled
                    ? document.getElementById('research-semantic-api-key').value
                    : '',
                images: researchImageAttachments.map(image => ({
                    name: image.name,
                    mime_type: image.mime_type,
                    data_url: image.data_url,
                })),
                async: true,
                generate_plots: true
            })
        });
        let data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || '研究流程执行失败');
        if (data.status === 'running') {
            let completed = false;
            for (let poll = 0; poll < 800; poll++) {
                await new Promise(resolve => setTimeout(resolve, 1500));
                const statusResponse = await fetch('/api/research/status');
                const statusData = await statusResponse.json();
                if (statusData.status === 'error') throw new Error(statusData.error || '研究任务执行失败');
                if (statusData.status === 'done') {
                    data = { success: true, result: statusData.result };
                    completed = true;
                    break;
                }
            }
            if (!completed) throw new Error('研究任务超过等待时限，请稍后重试');
        }
        renderResearchResult(data.result);
        resultBox.classList.remove('hidden');
        showToast('研究完成：已生成可审计数学证据包', 'success');
    } catch (error) {
        resultBox.innerHTML = `<div class="research-warning">研究未完成：${escapeHtml(error.message)}</div>`;
        resultBox.classList.remove('hidden');
        showToast('研究流程失败: ' + error.message, 'error');
    } finally {
        clearInterval(timer);
        progress.classList.add('hidden');
        button.disabled = false;
    }
}

async function clearResearchCache(button) {
    const original = button.textContent;
    button.disabled = true;
    button.textContent = '正在清理…';
    try {
        const response = await fetch('/api/research/cache', { method: 'DELETE' });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || '缓存清理失败');
        const cleanup = data.cleanup || {};
        showToast(`已清理 ${cleanup.deleted_files || 0} 个缓存文件；证据、报告和图表均保留`, 'success');
        button.textContent = '本次缓存已清理';
    } catch (error) {
        button.disabled = false;
        button.textContent = original;
        showToast(error.message, 'error');
    }
}

function formatResearchValue(value) {
    if (typeof value !== 'number') return escapeHtml(String(value ?? '-'));
    if (!Number.isFinite(value)) return '-';
    return Math.abs(value) >= 1000 ? value.toLocaleString() : Number(value.toPrecision(5)).toString();
}

function renderResearchResult(result) {
    const box = document.getElementById('research-result');
    const profiles = result.dataset_profiles || [];
    const relations = result.relationships || [];
    const interactions = result.interactions || [];
    const researchStatusLabels = {
        model_draft_ready: '数学草案已形成', needs_confirmation: '待符号与单位确认',
        not_applicable: '不适用', not_applicable_without_observations: '无观测数据时不适用',
        not_assessed: '未评估', needs_input: '需要补充输入', partial: '部分完成',
        ready: '已就绪', executed: '已执行', solver_ready: '求解器已就绪',
        partially_executed: '部分数值已执行', partially_ready: '部分关系可执行',
        complete: '完整', complete_with_gaps: '完整但有待绑定项',
        draft_only: '仅数学草案', runnable: '可运行', deferred: '已延后',
        blocked: '受阻', warning: '有条件通过', fail: '未通过',
        supported: '当前契约内支持', conditionally_supported: '有条件支持',
        rejected: '拒绝结论',
        machine_compiled: '题面确定性编译',
        needs_model_completion: '数学草案已形成，数值契约待确认',
        conceptual_model_compiled: '规范方程草案已编译',
        numerically_executable: '可执行数值模型',
        template_requires_binding: '规范形式待绑定', roles_bound: '角色已绑定',
        ready_to_compile: '可编译', partially_specified: '部分绑定',
        accepted: '候选全部通过', partially_accepted: '部分候选通过',
        no_accepted_relations: '候选均未通过', failed_safe: '失败并安全降级',
        pass: '通过', restricted: '限定范围内采用', unresolved: '待完成'
    };
    const researchRequirementLabels = {
        machine_readable_equations_or_algorithms: '可机器读取的方程或算法',
        verified_symbol_and_unit_bindings: '题面符号、单位与规范方程的最终核验',
        model_contract_confirmation: '模型契约确认'
    };
    const researchStatusText = value => researchStatusLabels[value] || value || '-';
    const researchRequirementText = value => researchRequirementLabels[value] || value;
    let html = '<div class="research-hero">';
    html += `<div><span class="research-kicker">自动研究已完成</span><h3>${escapeHtml(result.problem_analysis.model_class || '数学建模分析')}</h3><p>${escapeHtml(result.problem_analysis.model_description || '')}</p></div>`;
    html += `<div class="research-score">${result.problem_analysis.confidence || '-'}<small>% 题型识别置信度</small></div></div>`;

    html += '<div class="research-stats">';
    if (profiles.length) {
    html += `<div><strong>${profiles.length}</strong><span>数据集</span></div>`;
    html += `<div><strong>${relations.length}</strong><span>跨表关系</span></div>`;
    html += `<div><strong>${interactions.length}</strong><span>变量交互</span></div>`;
    } else {
        const mechanismPreview = ((result.specialized_results || {}).mechanistic_model || {});
        const graphPreview = (result.problem_analysis || {}).task_graph || [];
        html += `<div><strong>${graphPreview.length}</strong><span>子问题</span></div>`;
        html += `<div><strong>${graphPreview.filter(item => item.status === 'executed').length}</strong><span>已执行节点</span></div>`;
        html += `<div><strong>${(mechanismPreview.numerical_results || []).length}</strong><span>数值结果</span></div>`;
    }
    html += `<div><strong>${(result.charts || []).length}</strong><span>自动图表</span></div></div>`;

    const spec = result.mathematical_model_spec || {};
    const evidence = result.evidence_bundle || {};
    if (Object.keys(spec).length || Object.keys(evidence).length) {
        const overallStatus = evidence.overall_status || 'no_claims';
        const overallClass = overallStatus === 'contains_rejected_claims'
            ? 'research-fail'
            : (['empirical', 'conditional'].includes(overallStatus) ? 'research-safe' : 'research-risk');
        const integrity = (evidence.argument_integrity || {}).status;
        const integrityClass = integrity === 'pass' ? 'research-safe' : 'research-fail';
        const readinessTracks = spec.readiness_by_track || {};
        const supportedClaims = (evidence.claims || []).filter(claim => ['accepted_with_scope', 'restricted'].includes(claim.disposition)).length;
        const pendingClaims = (evidence.claims || []).filter(claim => claim.disposition === 'unresolved').length;
        html += '<details class="research-section" open><summary>数学规范与论证证据</summary>';
        html += `<div class="research-metrics"><span><small>机理数学结构</small><strong>${escapeHtml(researchStatusText(readinessTracks.mechanistic_structure || spec.readiness))}</strong></span><span><small>数值执行</small><strong>${escapeHtml(researchStatusText(readinessTracks.numerical_execution || 'not_assessed'))}</strong></span><span><small>观测数据建模</small><strong>${escapeHtml(researchStatusText(readinessTracks.observational_modeling || 'not_assessed'))}</strong></span><span><small>论证总状态</small><strong class="${overallClass}">${escapeHtml(evidence.overall_label || '尚无数值结论')}</strong></span><span><small>有据结论 / 待验证</small><strong>${supportedClaims} / ${pendingClaims}</strong></span><span><small>证据引用完整性</small><strong class="${integrityClass}">${escapeHtml(researchStatusText(integrity || '-'))}</strong></span></div>`;
        const roleLabels = { spatial_entities: '空间实体', stated_parameters: '题面参数', treatment: '处理变量', outcome: '结果变量', target: '目标变量', time: '时间变量' };
        const roles = Object.entries(spec.role_bindings || {}).map(([role, value]) => `${roleLabels[role] || role}=${value}`).join('；') || '尚无显式角色绑定';
        const pendingRequirements = (spec.missing_requirements || []).map(researchRequirementText).join('；') || '无';
        html += `<p class="hint"><strong>已绑定角色：</strong>${escapeHtml(roles)}<br><strong>未执行节点仍需：</strong>${escapeHtml(pendingRequirements)}<br><strong>静态矛盾：</strong>${escapeHtml((spec.contradictions || []).map(item => item.message).join('；') || '未发现')}</p>`;
        if ((evidence.claims || []).length) {
            html += '<h4>结论分级与处置</h4><div class="table-wrapper"><table class="data-table"><thead><tr><th>等级</th><th>结论</th><th>处置</th><th>适用边界</th></tr></thead><tbody>';
            evidence.claims.slice(0, 30).forEach(claim => {
                const claimClass = claim.disposition === 'rejected' ? 'research-fail' : (claim.disposition === 'unresolved' ? 'research-risk' : 'research-safe');
                html += `<tr><td><span class="${claimClass}">${escapeHtml(claim.label || '-')}</span></td><td>${escapeHtml(claim.statement || '-')}</td><td>${escapeHtml(researchStatusText(claim.disposition || '-'))}</td><td>${escapeHtml(claim.scope || '-')}</td></tr>`;
            });
            html += '</tbody></table></div>';
        }
        const criticalAssumptions = (evidence.assumption_ledger || []).filter(item => item.critical && item.status !== 'checked');
        if (criticalAssumptions.length) {
            html += '<h4>仍限制结论的关键假设</h4><div class="table-wrapper"><table class="data-table"><thead><tr><th>假设</th><th>状态</th><th>当前证据</th><th>补强方式</th></tr></thead><tbody>';
            criticalAssumptions.slice(0, 20).forEach(item => {
                const assumptionClass = item.status === 'failed' ? 'research-fail' : 'research-risk';
                html += `<tr><td>${escapeHtml(item.text || '-')}</td><td><span class="${assumptionClass}">${escapeHtml(item.status || '-')}</span></td><td>${escapeHtml(item.evidence || '-')}</td><td>${escapeHtml(item.falsification || '-')}</td></tr>`;
            });
            html += '</tbody></table></div>';
        }
        if ((evidence.model_tournament || []).length) {
            html += `<p class="hint"><strong>竞争模型：</strong>已记录 ${evidence.model_tournament.length} 组候选比较；被选模型仍须通过独立确认与反证，胜出不等于真实。</p>`;
        }
        html += '<p class="research-warning">论文写作 API 当前关闭。它在最后阶段只能改写获准结论，并必须保留假设、边界和反证。</p></details>';
    }

    const taskGraph = result.problem_analysis.task_graph || [];
    if (taskGraph.length) {
        html += '<details class="research-section" open><summary>多子问题执行图</summary><div class="table-wrapper"><table class="data-table"><thead><tr><th>节点</th><th>识别任务</th><th>状态</th><th>上游</th><th>证据与缺失条件</th></tr></thead><tbody>';
        taskGraph.forEach(node => {
            const dependencies = (node.depends_on || []).join('、') || '-';
            const missing = (node.missing_requirements || []).length ? `；缺少：${node.missing_requirements.map(researchRequirementText).join('、')}` : '';
            html += `<tr><td><strong>${escapeHtml(node.id || '-')}</strong><br><small>${escapeHtml(node.text || '')}</small></td><td>${escapeHtml(node.task_type || '-')}</td><td><span class="capability-status status-${escapeHtml(node.status || 'planned')}">${escapeHtml(researchStatusText(node.status || 'planned'))}</span></td><td>${escapeHtml(dependencies)}</td><td>${escapeHtml((node.evidence || '-') + missing)}</td></tr>`;
        });
        html += '</tbody></table></div><p class="hint">后续问题会显式引用上游结果；上游证据不足时，下游节点会标记为 blocked，而不是继续编造数值。</p></details>';
    }

    if (profiles.length) {
    html += '<details class="research-section" open><summary>数据集角色与结构</summary><div class="table-wrapper"><table class="data-table"><thead><tr><th>数据集</th><th>角色</th><th>规模</th><th>数值/类别/时间列</th><th>目标候选</th></tr></thead><tbody>';
    profiles.forEach(profile => {
        const candidates = (profile.target_candidates || []).slice(0, 2).map(item => `${item.column} (${Math.round(item.score * 100)}%)`).join('、') || '-';
        html += `<tr><td>${escapeHtml(profile.name)}</td><td><span class="research-role role-${escapeHtml(profile.role)}">${escapeHtml(profile.role)}</span></td><td>${Number(profile.source_rows).toLocaleString()} × ${profile.n_columns}</td><td>${profile.numeric_columns.length} / ${profile.categorical_columns.length} / ${profile.datetime_columns.length}</td><td>${escapeHtml(candidates)}</td></tr>`;
    });
    html += '</tbody></table></div></details>';

    html += '<details class="research-section" open><summary>自动发现的数据关系</summary>';
    if (relations.length) {
        html += '<div class="research-relation-list">';
        relations.slice(0, 20).forEach(relation => {
            const safety = relation.safe_to_join ? '<span class="research-safe">可安全关联</span>' : '<span class="research-risk">需先聚合</span>';
            html += `<div class="research-relation"><div><strong>${escapeHtml(relation.left_dataset)}.${escapeHtml(relation.left_key)}</strong><span> ↔ </span><strong>${escapeHtml(relation.right_dataset)}.${escapeHtml(relation.right_key)}</strong></div><div>${escapeHtml(relation.relationship)} · 置信度 ${relation.confidence}% · 值域覆盖 ${Math.round(relation.value_overlap * 100)}% ${safety}</div>${relation.warning ? `<small>${escapeHtml(relation.warning)}</small>` : ''}</div>`;
        });
        html += '</div>';
    } else {
        html += '<p class="hint">没有发现证据充分的跨表键；系统不会凭列位置强行拼接数据。</p>';
    }
    html += '</details>';
    html += '<details class="research-section" open><summary>跨数据集交互结论</summary>';
    if (interactions.length) {
        html += '<ol class="research-findings">';
        interactions.slice(0, 12).forEach(item => {
            const interval = item.confidence_interval ? `，区间=[${formatResearchValue(item.confidence_interval[0])}, ${formatResearchValue(item.confidence_interval[1])}]` : '';
            const qValue = item.q_value !== null && item.q_value !== undefined ? '，FDR q=' + formatResearchValue(item.q_value) : '';
            const conditional = item.conditional_strength !== null && item.conditional_strength !== undefined ? '，条件ρ=' + formatResearchValue(item.conditional_strength) : '';
            const stability = item.stability_score !== null && item.stability_score !== undefined ? '，稳定性=' + Math.round(item.stability_score * 100) + '%' : '';
            const significance = item.significant === true
                ? ' <span class="research-safe">FDR显著</span>'
                : (item.significant === false
                    ? ' <span class="research-risk">FDR不显著，仅探索</span>'
                    : ' <span class="research-warning">未检验显著性</span>');
            html += `<li><strong>${escapeHtml(item.method)}=${formatResearchValue(item.strength)}</strong> ${escapeHtml(item.interpretation)}${significance} <small>n=${item.sample_size}${item.p_value !== null && item.p_value !== undefined ? '，p=' + formatResearchValue(item.p_value) : ''}${qValue}${interval}${conditional}${stability}</small></li>`;
        });
        html += '</ol>';
    } else {
        html += '<p class="hint">当前阈值下没有发现稳定的跨表数值关系，报告中保留了非线性和时滞分析建议。</p>';
    }
    html += '</details>';
    }

    if (result.capability_report) {
        html += '<details class="research-section" open><summary>题型执行能力与待补充条件</summary><div class="table-wrapper"><table class="data-table"><thead><tr><th>识别任务</th><th>状态</th><th>执行证据</th><th>仍需条件</th></tr></thead><tbody>';
        (result.capability_report.tasks || []).forEach(item => {
            const requirement = item.requirement ? String(item.requirement).split('；').map(researchRequirementText).join('；') : '-';
            html += `<tr><td>${escapeHtml(item.task_type)}</td><td><span class="capability-status status-${escapeHtml(item.status)}">${escapeHtml(researchStatusText(item.status))}</span></td><td>${escapeHtml(item.evidence || '-')}</td><td>${escapeHtml(requirement)}</td></tr>`;
        });
        html += '</tbody></table></div><div class="robustness-guards">';
        (result.capability_report.robustness_guards || []).forEach(item => { html += `<span>✓ ${escapeHtml(item)}</span>`; });
        html += '</div></details>';
    }

    const specialized = result.specialized_results || {};
    const mechanismCandidate = specialized.mechanistic_model || null;
    const mechanismIrCandidate = (mechanismCandidate || {}).mathematical_ir || {};
    const hasMechanism = (mechanismCandidate || {}).presentation_scope !== 'internal_semantic_support' && Boolean(mechanismCandidate) && (
        ((mechanismCandidate.operator_graph || []).length > 0) ||
        ['entities', 'quantities', 'relations', 'objectives', 'constraints']
            .some(key => (mechanismIrCandidate[key] || []).length > 0) ||
        ![undefined, null, 'not_configured'].includes(
            (mechanismCandidate.semantic_model_compilation || {}).status
        )
    );
    const hasOtherSpecialized = Object.entries(specialized)
        .some(([key, value]) => key !== 'mechanistic_model' && Boolean(value));
    if (hasMechanism || hasOtherSpecialized) {
        html += '<details class="research-section" open><summary>专项数学分析</summary>';
        const dataCompilation = specialized.mathematical_data_compilation || null;
        if (dataCompilation) {
            const contract = dataCompilation.contract || {};
            const compilationSummary = dataCompilation.summary || {};
            const compilationClass = dataCompilation.status === 'contradicted'
                ? 'research-fail'
                : (dataCompilation.status === 'assessed' ? 'research-safe' : 'research-risk');
            html += `<h4>数学数据多视图编译 <span class="${compilationClass}">${escapeHtml(dataCompilation.status || '-')}</span></h4>`;
            html += `<div class="research-metrics"><span><small>主估计数据集</small><strong>${escapeHtml(dataCompilation.dataset || '-')}</strong></span><span><small>目标</small><strong>${escapeHtml(contract.target || '未绑定')}</strong></span><span><small>观测粒度</small><strong>${escapeHtml((contract.observed_grain || []).join(' × ') || '未验证')}</strong><small>${escapeHtml(contract.grain_status || '-')}</small></span><span><small>审计行数</small><strong>${Number(compilationSummary.audited_rows || 0).toLocaleString()} / ${Number(compilationSummary.source_rows || 0).toLocaleString()}</strong></span><span><small>编译耗时</small><strong>${Number((compilationSummary.timing_ms || {}).multi_table_total ?? (compilationSummary.timing_ms || {}).total ?? 0).toFixed(1)} ms</strong></span><span><small>候选通过</small><strong>${compilationSummary.admissible_views || 0} / ${compilationSummary.candidate_views || 0}</strong></span><span><small>跨表阻断</small><strong class="${compilationSummary.blocked_cross_dataset_contracts ? 'research-fail' : 'research-safe'}">${compilationSummary.blocked_cross_dataset_contracts || 0}</strong></span><span><small>方向翻转</small><strong class="${compilationSummary.direction_reversals ? 'research-fail' : 'research-safe'}">${compilationSummary.direction_reversals || 0}</strong></span></div>`;
            html += `<p class="hint"><strong>估计对象：</strong>${escapeHtml(contract.estimand || '-')}</p>`;
            const reversedRelationships = ((dataCompilation.conclusion_stress || {}).relationships || [])
                .filter(item => item.status === 'contradicted');
            if (reversedRelationships.length) {
                html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>被反证关系</th><th>全局ρ / 95%区间</th><th>全局FDR q</th><th>翻转视图</th><th>处置</th></tr></thead><tbody>';
                reversedRelationships.slice(0, 20).forEach(item => {
                    const global = (item.contexts || []).find(context => context.view === 'global_complete_case');
                    const interval = (global?.confidence_interval_95 || []).join(', ');
                    html += `<tr><td>${escapeHtml(item.predictor)} → ${escapeHtml(item.target)}</td><td>${global?.rho ?? '-'}<br><small>[${escapeHtml(interval || '-')} ]</small></td><td>${item.global_fdr_q ?? '-'}</td><td>${escapeHtml((item.direction_flips || []).map(flip => flip.against).join('、') || '-')}</td><td class="research-fail">禁止写成稳定规律</td></tr>`;
                });
                html += '</tbody></table></div>';
            }
            const crossContracts = dataCompilation.cross_dataset_contracts || [];
            if (crossContracts.length) {
                html += '<h4>跨表粒度与连接契约</h4><div class="table-wrapper"><table class="data-table"><thead><tr><th>数据表</th><th>复合键</th><th>基数</th><th>膨胀</th><th>时间对齐</th><th>处置</th></tr></thead><tbody>';
                crossContracts.slice(0, 30).forEach(item => {
                    const keys = (item.key_pairs || []).map(pair => `${pair.left}↔${pair.right}`).join('、') || '未验证';
                    const contractClass = item.status === 'blocked' ? 'research-fail'
                        : (item.status === 'admissible' ? 'research-safe' : 'research-risk');
                    const reaudit = item.full_cardinality_reaudit_required ? '；需全表复审' : '';
                    html += `<tr><td>${escapeHtml(item.left_dataset)} ↔ ${escapeHtml(item.right_dataset)}</td><td>${escapeHtml(keys)}</td><td>${escapeHtml(item.relationship || '-')}</td><td>${item.estimated_expansion ?? '-'}</td><td>${item.point_in_time_required ? '必须 point-in-time' : '普通键对齐'}</td><td><span class="${contractClass}">${escapeHtml(item.status || '-')}</span><br><small>${escapeHtml((item.combined_additive_analysis || item.evidence || '-') + reaudit)}</small></td></tr>`;
                });
                html += '</tbody></table></div>';
            }
            (dataCompilation.findings || []).forEach(item => {
                html += `<p class="${['contradicted', 'blocked'].includes(item.level) ? 'research-warning' : 'hint'}">${escapeHtml(item.message || '')} ${escapeHtml(item.action || '')}</p>`;
            });
        }
        const mechanism = hasMechanism ? mechanismCandidate : null;
        if (mechanism) {
            const mathIr = mechanism.mathematical_ir || {};
            const compiler = mechanism.compiler_plan || {};
            const modelDraft = mechanism.model_draft || {};
            const audit = mechanism.credibility_audit || {};
            const semanticModel = mechanism.semantic_model_compilation || {};
            const fourLayer = mechanism.four_layer_pipeline || {};
            const semanticContract = fourLayer.semantic_contract || {};
            const unifiedIr = fourLayer.mathematical_ir || {};
            const solverPlan = fourLayer.solver_plan || {};
            const independentAudit = fourLayer.independent_audit || {};
            const requirementLabels = {
                machine_readable_equations_or_algorithms: '可机器读取的方程或算法',
                verified_symbol_and_unit_bindings: '题面符号、单位与规范方程的最终核验',
                decision_variables: '决策变量', objective_function: '目标函数',
                constraints_and_bounds: '约束与变量边界', state_variables: '状态变量',
                initial_conditions: '初始条件', boundary_conditions: '边界条件',
                dynamics_or_transition_rule: '动力学或状态转移规则',
                geometry_definition: '几何对象与距离定义', event_definition: '事件判定定义',
                testable_hypothesis: '可检验假设', variables_and_sampling_unit: '变量与样本单位'
            };
            const stageLabels = {
                problem_decomposition: '问题分解', quantity_and_entity_extraction: '实体与显式量抽取',
                operator_composition: '通用算子组合', canonical_equation_draft: '规范方程草案',
                operator_selection: '算子选择', binding: '角色绑定'
            };
            const humanRequirement = value => requirementLabels[value] || stageLabels[value] || researchRequirementText(value);
            const auditClass = audit.status === 'pass' ? 'research-safe' : (audit.status === 'fail' ? 'research-fail' : 'research-risk');
            if (semanticModel.status && semanticModel.status !== 'not_configured') {
                const semanticConfig = semanticModel.configuration || {};
                const semanticClass = ['accepted', 'partially_accepted'].includes(semanticModel.status)
                    ? 'research-safe'
                    : (semanticModel.status === 'failed_safe' ? 'research-risk' : 'research-warning');
                html += `<h4>受约束语义模型编译 <span class="${semanticClass}">${escapeHtml(researchStatusText(semanticModel.status))}</span></h4>`;
                html += `<div class="research-metrics"><span><small>模型后端</small><strong>${escapeHtml(semanticConfig.provider || '-')}</strong></span><span><small>模型</small><strong>${escapeHtml(semanticConfig.model_name || '-')}</strong></span><span><small>接受 / 延后</small><strong>${semanticModel.accepted_count || 0} / ${semanticModel.deferred_count || 0}</strong></span><span><small>密钥写入产物</small><strong>否</strong></span></div>`;
                html += '<p class="hint">模型只提出候选 IR；逐字段题面引文、数值溯源和确定性契约校验全部通过后才能执行。</p>';
                if (semanticModel.error) {
                    html += `<p class="research-warning"><strong>已安全降级：</strong>${escapeHtml(semanticModel.error)}</p>`;
                }
                (semanticModel.deferred_proposals || []).slice(0, 12).forEach(item => {
                    html += `<p class="research-warning"><strong>延后 ${escapeHtml(item.id || String(item.index ?? '-'))}：</strong>${escapeHtml((item.errors || []).join('；') || '未通过语义证据门')}</p>`;
                });
            }
            if (fourLayer.schema_version) {
                const planBudget = solverPlan.budget_summary || {};
                const auditCoverage = independentAudit.coverage || {};
                const structureCatalog = unifiedIr.structure_catalog || [];
                const candidateStructures = semanticContract.candidate_structures || [];
                const implementedStructures = structureCatalog.filter(item => item.execution_status === 'implemented').length;
                const layerAuditClass = independentAudit.status === 'pass' ? 'research-safe' : (independentAudit.status === 'fail' ? 'research-fail' : 'research-risk');
                html += `<h4>四层数学建模流水线 <span class="${layerAuditClass}">${escapeHtml(independentAudit.status || '-')}</span></h4>`;
                html += `<div class="research-metrics"><span><small>①题意契约</small><strong>${escapeHtml(researchStatusText(semanticContract.status || '-'))}</strong></span><span><small>②统一 IR</small><strong>${escapeHtml(researchStatusText(unifiedIr.status || '-'))}</strong></span><span><small>③结构选解</small><strong>${escapeHtml(researchStatusText(solverPlan.status || '-'))}</strong></span><span><small>④独立审计</small><strong>${escapeHtml(independentAudit.status || '-')}</strong></span><span><small>可运行 / 延后</small><strong>${planBudget.runnable_nodes || 0} / ${planBudget.deferred_nodes || 0}</strong></span><span><small>仅语义候选</small><strong>${((unifiedIr.validation || {}).semantic_candidates) || 0}</strong></span><span><small>审计覆盖</small><strong>${auditCoverage.audited_results || 0} / ${auditCoverage.executed_results || 0}</strong></span><span><small>数学结构目录</small><strong>${structureCatalog.length}</strong></span><span><small>已实现 / 仅识别</small><strong>${implementedStructures} / ${structureCatalog.length - implementedStructures}</strong></span></div>`;
                html += '<p class="hint">求解器只按数学形式选择；单节点失败会被隔离，数值成功不会自动等同于模型正确。</p>';
                html += `<p class="hint"><strong>题面结构候选：</strong>${escapeHtml(candidateStructures.map(item => item.key).join('、') || '未可靠识别')}。关键词命中只生成候选，不会直接放行数值执行。</p>`;
                if ((solverPlan.nodes || []).length) {
                    html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>统一 IR 节点</th><th>数学形式</th><th>求解器族</th><th>状态</th><th>预算</th></tr></thead><tbody>';
                    solverPlan.nodes.slice(0, 40).forEach(node => {
                        const budget = node.resource_budget || {};
                        const budgetText = `变量≤${budget.max_variables ?? '-'}；评估≤${budget.max_evaluations ?? '-'}；软墙钟预算 ${budget.wall_time_budget_seconds ?? '-'}s`;
                        html += `<tr><td>${escapeHtml(node.ir_node_id || '-')}</td><td>${escapeHtml(node.mathematical_form || '-')}</td><td>${escapeHtml(node.solver_family || '-')}</td><td>${escapeHtml(researchStatusText(node.status || '-'))}</td><td>${escapeHtml(budgetText)}</td></tr>`;
                    });
                    html += '</tbody></table></div>';
                }
                if ((unifiedIr.deferred_semantic_relations || []).length) {
                    html += '<details><summary>未进入求解计划的语义候选</summary><div class="table-wrapper"><table class="data-table"><thead><tr><th>关系</th><th>类型</th><th>原因</th><th>原文证据</th></tr></thead><tbody>';
                    unifiedIr.deferred_semantic_relations.slice(0, 30).forEach(item => {
                        html += `<tr><td>${escapeHtml(item.relation_id || '-')}</td><td>${escapeHtml(item.kind || '-')}</td><td>${escapeHtml(item.reason || '-')}</td><td>${escapeHtml(item.source_text || '-')}</td></tr>`;
                    });
                    html += '</tbody></table></div></details>';
                }
                if (structureCatalog.length) {
                    html += '<details><summary>通用数学结构能力矩阵</summary><div class="table-wrapper"><table class="data-table"><thead><tr><th>数学结构</th><th>家族</th><th>后端状态</th><th>求解器</th><th>结构化契约必需字段</th></tr></thead><tbody>';
                    structureCatalog.forEach(item => {
                        const solver = item.solver || {};
                        const status = item.execution_status === 'implemented' ? '已实现' : '仅识别';
                        html += `<tr><td>${escapeHtml(item.key || '-')}</td><td>${escapeHtml(item.family || '-')}</td><td>${escapeHtml(status)}</td><td>${escapeHtml(solver.solver_family || '-')}</td><td>${escapeHtml((item.required_contract_fields || []).join('、') || '-')}</td></tr>`;
                    });
                    html += '</tbody></table></div></details>';
                }
                if ((independentAudit.execution_failures || []).length) {
                    html += `<p class="research-warning"><strong>已隔离失败节点：</strong>${escapeHtml(JSON.stringify(independentAudit.execution_failures))}</p>`;
                }
            }
            html += `<h4>纯题面通用数学 IR <span class="${auditClass}">${escapeHtml(audit.label || '-')}</span></h4>`;
            html += `<div class="research-metrics"><span><small>执行状态</small><strong>${escapeHtml(researchStatusText(mechanism.execution_status || '-'))}</strong></span><span><small>模型草案</small><strong>${escapeHtml(researchStatusText(modelDraft.status || '-'))}</strong></span><span><small>实体</small><strong>${(mathIr.entities || []).length}</strong></span><span><small>显式量</small><strong>${(mathIr.quantities || []).length}</strong></span><span><small>数学关系</small><strong>${(mathIr.relations || []).length}</strong></span><span><small>通用算子</small><strong>${(mechanism.operator_graph || []).length}</strong></span><span><small>规范方程</small><strong>${(modelDraft.equations || []).length}</strong></span></div>`;
            if ((modelDraft.completed_stages || []).length) {
                html += `<p class="hint"><strong>已经完成：</strong>${escapeHtml(modelDraft.completed_stages.map(humanRequirement).join(' → '))}</p>`;
            }
            if ((mechanism.operator_graph || []).length) {
                html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>算子</th><th>类别</th><th>状态</th><th>求解路线</th><th>未绑定角色</th></tr></thead><tbody>';
                mechanism.operator_graph.slice(0, 30).forEach(node => {
                    html += `<tr><td>${escapeHtml(node.key || '-')}</td><td>${escapeHtml(node.category || '-')}</td><td>${escapeHtml(researchStatusText(node.status || '-'))}</td><td>${escapeHtml(node.solver_route || '-')}</td><td>${escapeHtml((node.missing_bindings || []).map(humanRequirement).join('、') || '-')}</td></tr>`;
                });
                html += '</tbody></table></div>';
            }
            if ((modelDraft.equations || []).length) {
                html += '<h4>规范方程草案</h4><div class="table-wrapper"><table class="data-table"><thead><tr><th>通用算子</th><th>规范形式</th><th>状态</th><th>仍需绑定</th></tr></thead><tbody>';
                modelDraft.equations.slice(0, 30).forEach(equation => {
                    const missing = (equation.missing_bindings || []).map(humanRequirement).join('、') || '无';
                    html += `<tr><td>${escapeHtml(equation.operator || '-')}</td><td><code>${escapeHtml(equation.expression || '-')}</code></td><td>${escapeHtml(researchStatusText(equation.status || '-'))}</td><td>${escapeHtml(missing)}</td></tr>`;
                });
                html += '</tbody></table></div>';
            }
            if ((modelDraft.assumption_questions || []).length) {
                html += '<h4>数值求解前必须回答的假设问题</h4><ul>';
                modelDraft.assumption_questions.forEach(question => { html += `<li>${escapeHtml(question)}</li>`; });
                html += '</ul>';
            }
            const blockers = compiler.blocked_by || mechanism.missing_requirements || [];
            if (blockers.length) {
                html += `<p class="research-warning"><strong>数学草案已建立，但数值求解仍待补：</strong>${escapeHtml(blockers.map(humanRequirement).join('；'))}。系统不会把题面散文直接当代码执行。</p>`;
            }
            (mechanism.numerical_results || []).slice(0, 6).forEach((numerical, index) => {
                const convergence = numerical.convergence || {};
                const convergenceClass = convergence.status === 'pass' ? 'research-safe' : 'research-fail';
                const resultAudit = numerical.independent_audit || {};
                const resultAuditClass = resultAudit.status === 'pass' ? 'research-safe' : (resultAudit.status === 'fail' ? 'research-fail' : 'research-risk');
                html += `<h4>通用数值执行 ${index + 1} · ${escapeHtml(numerical.kind || '-')} <span class="${convergenceClass}">${escapeHtml(convergence.status || '-')}</span> <span class="${resultAuditClass}">独立审计 ${escapeHtml(resultAudit.grade || 'not_assessed')}</span></h4>`;
                if ((resultAudit.false_confidence_flags || []).length) {
                    html += `<p class="research-warning"><strong>“似对非对”风险标记：</strong>${escapeHtml(resultAudit.false_confidence_flags.join('、'))}。${escapeHtml(resultAudit.decision || '')}</p>`;
                }
                if (numerical.kind === 'kinematic_visibility_event') {
                    const range = numerical.semantic_duration_range || [];
                    html += `<div class="research-metrics"><span><small>有效遮蔽时长</small><strong>${formatResearchValue(numerical.duration)} s</strong></span><span><small>有效区间</small><strong>${escapeHtml(JSON.stringify(numerical.effective_intervals || []))}</strong></span><span><small>起爆时刻</small><strong>${formatResearchValue(numerical.activation_time)} s</strong></span><span><small>语义分支范围</small><strong>[${range.map(formatResearchValue).join(', ')}] s</strong></span></div>`;
                    html += `<p class="hint"><strong>投放点：</strong>${escapeHtml(JSON.stringify(numerical.release_point || []))}<br><strong>起爆点：</strong>${escapeHtml(JSON.stringify(numerical.activation_point || []))}<br><strong>网格加密最大时长差：</strong>${formatResearchValue(convergence.maximum_duration_difference)} s<br><strong>可信度边界：</strong>${escapeHtml((numerical.credibility_audit || {}).decision || '-')}</p>`;
                } else if (numerical.kind === 'kinematic_visibility_optimization_solution') {
                    const implementation = numerical.implementation_candidate || {};
                    const feedback = numerical.feedback_optimization || {};
                    html += `<div class="research-metrics"><span><small>最佳可行时长</small><strong>${formatResearchValue(numerical.duration)} s</strong></span><span><small>多起点近优差</small><strong>${formatResearchValue((numerical.multistart_relative_spread || 0) * 100)}%</strong></span><span><small>扰动最大下降</small><strong>${formatResearchValue((numerical.maximum_relative_sensitivity_drop || 0) * 100)}%</strong></span><span><small>有限候选</small><strong>${numerical.successful_starts || 0}</strong></span></div>`;
                    html += `<p class="hint"><strong>决策参数：</strong>${escapeHtml(JSON.stringify(numerical.solution || {}))}<br><strong>投放点：</strong>${escapeHtml(JSON.stringify(numerical.release_point || []))}<br><strong>激活点：</strong>${escapeHtml(JSON.stringify(numerical.activation_point || []))}<br><strong>有效区间：</strong>${escapeHtml(JSON.stringify(numerical.effective_intervals || []))}<br><strong>99%近优范围：</strong>${escapeHtml(JSON.stringify(numerical.one_at_a_time_99pct_ranges || {}))}<br><strong>舍入实施方案：</strong>${escapeHtml(JSON.stringify(implementation.solution || {}))}，时长 ${formatResearchValue(implementation.duration)} s，${implementation.accepted ? '可采用' : '舍入损失较大'}<br><strong>相对基线改善：</strong>${feedback.relative_gain == null ? '-' : formatResearchValue(feedback.relative_gain * 100) + '%'}<br><strong>可信度边界：</strong>${escapeHtml((numerical.credibility_audit || {}).decision || '-')}</p>`;
                } else {
                    html += `<p class="hint"><strong>求解器：</strong>${escapeHtml(numerical.solver || '-')}<br><strong>状态摘要：</strong>${escapeHtml(JSON.stringify(numerical.summary || {}))}<br><strong>容差复算相对差：</strong>${formatResearchValue(convergence.relative_tolerance_comparison)}</p>`;
                }
            });
            html += `<p class="hint">${escapeHtml(audit.decision || 'IR 编译完成不等于数值答案；数值收敛也不等于机理真实。')}</p>`;
        }
        const hierarchicalSales = specialized.hierarchical_distribution || specialized.hierarchical_sales;
        if (hierarchicalSales) {
            const hierarchyAudit = hierarchicalSales.credibility_audit || {};
            const concentration = hierarchicalSales.concentration || {};
            const parentDimension = hierarchicalSales.parent_dimension || hierarchicalSales.category_column || '上层维度';
            const childDimension = hierarchicalSales.child_dimension || hierarchicalSales.item_column || '下层维度';
            html += `<h4>${escapeHtml(parentDimension)}—${escapeHtml(childDimension)}层级分布与剩余联动 <span class="${hierarchyAudit.status === 'pass' ? 'research-safe' : 'research-risk'}">${escapeHtml(hierarchyAudit.label || '-')}</span></h4>`;
            html += `<div class="research-metrics"><span><small>聚合源行</small><strong>${Number(hierarchicalSales.source_rows_aggregated || 0).toLocaleString()}</strong></span><span><small>日×上层×下层</small><strong>${Number(hierarchicalSales.daily_item_rows || 0).toLocaleString()}</strong></span><span><small>下层对象数</small><strong>${concentration.child_count || concentration.item_count || 0}</strong></span><span><small>下层 HHI</small><strong>${formatResearchValue(concentration.hhi)}</strong></span><span><small>前20份额</small><strong>${formatResearchValue((concentration.top_20_share || 0) * 100)}%</strong></span></div>`;
            html += `<div class="table-wrapper"><table class="data-table"><thead><tr><th>${escapeHtml(parentDimension)}</th><th>总量</th><th>日均</th><th>日标准差</th><th>变异系数</th><th>P10 / 中位数 / P90</th></tr></thead><tbody>`;
            (hierarchicalSales.parent_summary || hierarchicalSales.category_summary || []).slice(0, 50).forEach(row => {
                const category = row[parentDimension] ?? '-';
                html += `<tr><td>${escapeHtml(category)}</td><td>${formatResearchValue(row.total)}</td><td>${formatResearchValue(row.daily_mean)}</td><td>${formatResearchValue(row.daily_std)}</td><td>${formatResearchValue(row.coefficient_of_variation)}</td><td>${formatResearchValue(row.q10)} / ${formatResearchValue(row.median)} / ${formatResearchValue(row.q90)}</td></tr>`;
            });
            html += '</tbody></table></div>';
            const associationRows = [
                ...(hierarchicalSales.parent_associations || hierarchicalSales.category_associations || []).map(item => ({...item, level: parentDimension})),
                ...(hierarchicalSales.child_associations || hierarchicalSales.item_associations || []).map(item => ({...item, level: childDimension}))
            ].slice(0, 60);
            html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>层级</th><th>对象 A</th><th>对象 B</th><th>残差 ρ</th><th>FDR q</th><th>处置</th></tr></thead><tbody>';
            associationRows.forEach(row => {
                html += `<tr><td>${row.level}</td><td>${escapeHtml(row.left)}</td><td>${escapeHtml(row.right)}</td><td>${formatResearchValue(row.residual_spearman)}</td><td>${formatResearchValue(row.q_value)}</td><td>${row.significant ? '<span class="research-safe">FDR显著</span>' : '<span class="research-risk">仅探索</span>'}</td></tr>`;
            });
            html += `</tbody></table></div><p class="hint">${escapeHtml(hierarchyAudit.decision || '')} ${escapeHtml(hierarchicalSales.note || '')}</p>`;
        }
        const groupedForecasts = (specialized.grouped_forecasts || []).length
            ? specialized.grouped_forecasts
            : (specialized.grouped_forecast ? [specialized.grouped_forecast] : []);
        groupedForecasts.forEach(groupedForecast => {
            const groupedMetrics = groupedForecast.metrics || {};
            const groupedAudit = groupedForecast.credibility_audit || {};
            const groupedClass = groupedAudit.status === 'pass' ? 'research-safe' : 'research-risk';
            html += `<h4>分组时间粒度预测 · ${escapeHtml(groupedForecast.requested_grain || 'group')} <span class="${groupedClass}">${escapeHtml(groupedAudit.label || '-')}</span></h4>`;
            html += `<div class="research-metrics"><span><small>编译粒度</small><strong>日 × ${escapeHtml(groupedForecast.group_column || '-')}</strong></span><span><small>聚合源行</small><strong>${Number(groupedForecast.source_rows_aggregated || 0).toLocaleString()}</strong></span><span><small>预测组数</small><strong>${groupedForecast.groups_forecast || 0}</strong></span><span><small>预测天数</small><strong>${groupedForecast.horizon_days || 0}</strong></span><span><small>末段RMSE</small><strong>${formatResearchValue(groupedMetrics.terminal_block_rmse)}</strong></span><span><small>季节基线RMSE</small><strong>${formatResearchValue(groupedMetrics.seasonal_naive_rmse)}</strong></span></div>`;
            html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>组</th><th>日期</th><th>点预测</th><th>90%区间</th><th>模型</th></tr></thead><tbody>';
            (groupedForecast.forecasts || []).slice(0, 500).forEach(row => {
                html += `<tr><td>${escapeHtml(row.group || '-')}</td><td>${escapeHtml(row.date || '-')}</td><td>${formatResearchValue(row.forecast)}</td><td>[${formatResearchValue(row.lower_90)}, ${formatResearchValue(row.upper_90)}]</td><td>${escapeHtml(row.selected_model || '-')}</td></tr>`;
            });
            html += `</tbody></table></div><p class="hint">${escapeHtml(groupedAudit.decision || '')} ${escapeHtml(groupedForecast.note || '')}</p>`;
        });
        const prescriptiveDecisions = (specialized.prescriptive_decisions || []).length
            ? specialized.prescriptive_decisions
            : (specialized.prescriptive_decision ? [specialized.prescriptive_decision] : []);
        prescriptiveDecisions.forEach(prescriptive => {
            const decisionAudit = prescriptive.credibility_audit || {};
            html += `<h4>预测—补货—定价组合决策 · ${escapeHtml(prescriptive.requested_grain || 'group')} <span class="research-risk">${escapeHtml(decisionAudit.label || '-')}</span></h4>`;
            const parentCoverage = prescriptive.aggregate_parent_demand_coverage;
            html += `<div class="research-metrics"><span><small>数学形式</small><strong>${escapeHtml(prescriptive.mathematical_form || '-')}</strong></span><span><small>决策数</small><strong>${prescriptive.decision_count || 0}</strong></span><span><small>允许调价</small><strong>${prescriptive.price_decision_count || 0}</strong></span><span><small>保持参考价</small><strong>${prescriptive.held_price_count || 0}</strong></span><span><small>数量边界</small><strong>${escapeHtml(JSON.stringify(prescriptive.selection_bounds || '-'))}</strong></span><span><small>最小陈列</small><strong>${formatResearchValue(prescriptive.minimum_display)}</strong></span><span><small>品类需求覆盖</small><strong>${parentCoverage == null ? '-' : `${formatResearchValue(parentCoverage * 100)}%`}</strong></span><span><small>最小缺口验证</small><strong>${prescriptive.hierarchical_lexicographic_verified ? '通过' : '否/不适用'}</strong></span><span><small>成本覆盖</small><strong>${Math.round((prescriptive.cost_coverage || 0) * 100)}%</strong></span><span><small>损耗覆盖</small><strong>${Math.round((prescriptive.loss_coverage || 0) * 100)}%</strong></span></div>`;
            html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>组</th><th>日期</th><th>需求</th><th>建议补货</th><th>补货90%范围</th><th>建议价格</th><th>单位成本</th><th>期望收益</th></tr></thead><tbody>';
            (prescriptive.decision_rows || []).slice(0, 500).forEach(row => {
                html += `<tr><td>${escapeHtml(row.group || '-')}</td><td>${escapeHtml(row.date || '-')}</td><td>${formatResearchValue(row.forecast_demand)}</td><td>${formatResearchValue(row.replenishment)}</td><td>[${formatResearchValue(row.lower_replenishment_90)}, ${formatResearchValue(row.upper_replenishment_90)}]</td><td>${formatResearchValue(row.price)}</td><td>${formatResearchValue(row.unit_cost)}</td><td>${formatResearchValue(row.payoff)}</td></tr>`;
            });
            html += '</tbody></table></div>';
            const riskStress = prescriptive.risk_aware_stress_test || null;
            if (riskStress) {
                const nominalRisk = riskStress.nominal_selection || {};
                const robustRisk = riskStress.risk_aware_selection || {};
                html += `<h5>预测区间情景与下行风险审计</h5><p class="hint">压力权重 ${escapeHtml(JSON.stringify(riskStress.scenario_weights || {}))} 不是经校准的概率；目标为 50% 压力加权期望 + 50% 的 75% 下尾 CVaR。改变决策单元数：${riskStress.changed_decision_unit_count || 0}；${escapeHtml(riskStress.decision || '')}</p>`;
                html += '<p class="research-warning">压力收益暂按未售出残值为 0、缺货不另计信誉/机会惩罚；存在折价、报废、库存结转或缺货损失时必须重算。</p>';
                html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>方案</th><th>压力加权期望</th><th>最坏情景</th><th>下尾CVaR</th><th>风险调整收益</th><th>采用</th></tr></thead><tbody>';
                html += `<tr><td>名义方案</td><td>${formatResearchValue(nominalRisk.stress_weighted_expected_utility)}</td><td>${formatResearchValue(nominalRisk.worst_case_utility)}</td><td>${formatResearchValue(nominalRisk.lower_tail_cvar)}</td><td>${formatResearchValue(nominalRisk.risk_adjusted_utility)}</td><td>${riskStress.adopted ? '否' : '是'}</td></tr>`;
                html += `<tr><td>风险感知候选</td><td>${formatResearchValue(robustRisk.stress_weighted_expected_utility)}</td><td>${formatResearchValue(robustRisk.worst_case_utility)}</td><td>${formatResearchValue(robustRisk.lower_tail_cvar)}</td><td>${formatResearchValue(robustRisk.risk_adjusted_utility)}</td><td>${riskStress.adopted ? '是' : '仅压力测试'}</td></tr>`;
                html += '</tbody></table></div>';
            }
            const coverageRows = prescriptive.hierarchical_demand_coverage || [];
            if (coverageRows.length) {
                html += '<h5>上层品类需求覆盖审计</h5><div class="table-wrapper"><table class="data-table"><thead><tr><th>品类</th><th>日期</th><th>预测需求</th><th>入选单品需求</th><th>缺口</th><th>覆盖率</th></tr></thead><tbody>';
                coverageRows.slice(0, 200).forEach(row => {
                    html += `<tr><td>${escapeHtml(row.parent_group || '-')}</td><td>${escapeHtml(row.date || '-')}</td><td>${formatResearchValue(row.target_demand)}</td><td>${formatResearchValue(row.selected_item_demand)}</td><td>${formatResearchValue(row.shortage)}</td><td>${formatResearchValue((row.coverage_ratio || 0) * 100)}%</td></tr>`;
                });
                html += '</tbody></table></div>';
            }
            const costPlusRows = prescriptive.cost_plus_pricing_relationship || [];
            if (costPlusRows.length) {
                html += `<h5>成本加成率—销量关系审计（观察性）</h5><p class="hint">售价来自 ${escapeHtml(prescriptive.dataset || '-')}，成本来自 ${escapeHtml(`${prescriptive.cost_dataset || '-'}.${prescriptive.cost_column || '-'}`)}。按日期×品类对齐并去除线性趋势与星期效应；只有 BH-FDR q≤0.05 且前后半段同号才标记为稳定。</p>`;
                html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>品类</th><th>对齐天数</th><th>中位加成率</th><th>残差相关</th><th>p值</th><th>FDR q值</th><th>前半/后半</th><th>判定</th></tr></thead><tbody>';
                costPlusRows.forEach(row => {
                    const first = row.first_half_spearman == null ? '-' : formatResearchValue(row.first_half_spearman);
                    const second = row.second_half_spearman == null ? '-' : formatResearchValue(row.second_half_spearman);
                    html += `<tr><td>${escapeHtml(row.group || '-')}</td><td>${row.n_aligned_days || 0}</td><td>${formatResearchValue((row.median_markup_rate || 0) * 100)}%</td><td>${formatResearchValue(row.residual_spearman)}</td><td>${formatResearchValue(row.p_value)}</td><td>${formatResearchValue(row.q_value)}</td><td>${first}/${second}</td><td>${row.significant ? '<span class="research-safe">FDR显著且方向稳定</span>' : '<span class="research-risk">未通过联合门</span>'}</td></tr>`;
                });
                html += '</tbody></table></div><p class="research-warning">这是观察性关系，不是调价因果效应；品类成本采用当日单品批发价中位数近似。</p>';
            }
            html += `<p class="research-warning">${escapeHtml(decisionAudit.decision || '')}</p><p class="hint">${escapeHtml(prescriptive.note || '')}</p>`;
        });
        const dataRequirements = specialized.data_requirements;
        if (dataRequirements) {
            html += `<h4>数据需求与可识别性审计 <span class="research-safe">${escapeHtml(researchStatusText(dataRequirements.status || 'executed'))}</span></h4>`;
            html += `<div class="research-metrics"><span><small>已审计数据集</small><strong>${dataRequirements.observed_dataset_count || 0}</strong></span><span><small>已审计字段</small><strong>${dataRequirements.observed_column_count || 0}</strong></span><span><small>关系证据</small><strong>${dataRequirements.relationship_evidence_count || 0}</strong></span><span><small>建议项</small><strong>${(dataRequirements.recommendations || []).length}</strong></span></div>`;
            html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>优先级</th><th>应补数据角色</th><th>为什么需要</th><th>采集设计</th><th>支持任务</th></tr></thead><tbody>';
            (dataRequirements.recommendations || []).forEach(item => {
                html += `<tr><td>${escapeHtml(item.priority || '-')}</td><td>${escapeHtml(item.data_role || '-')}</td><td>${escapeHtml(item.reason || '-')}</td><td>${escapeHtml(item.collection_design || '-')}</td><td>${escapeHtml((item.supports_tasks || []).join('、') || '-')}</td></tr>`;
            });
            html += `</tbody></table></div><p class="hint">${escapeHtml(dataRequirements.note || '')}</p>`;
        }
        const optimization = specialized.optimization;
        if (optimization) {
            const audit = optimization.credibility_audit || {};
            const auditClass = audit.status === 'pass' ? 'research-safe' : (audit.status === 'fail' ? 'research-fail' : 'research-risk');
            html += `<h4>显式连续线性优化 · HiGHS <span class="${auditClass}">${escapeHtml(audit.label || '-')}</span></h4>`;
            if (optimization.solver_success) {
                html += `<div class="research-metrics"><span><small>目标值</small><strong>${formatResearchValue(optimization.objective_value)}</strong></span><span><small>最大约束违反</small><strong>${formatResearchValue(optimization.maximum_constraint_violation)}</strong></span><span><small>最大KKT残差</small><strong>${formatResearchValue((optimization.optimality_certificate || {}).maximum_kkt_residual)}</strong></span><span><small>扰动成功次数</small><strong>${(optimization.sensitivity || {}).successful_runs || 0}</strong></span><span><small>中位方案变化</small><strong>${formatResearchValue((optimization.sensitivity || {}).median_relative_solution_shift)}</strong></span></div>`;
                html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>决策变量</th><th>最优值</th><th>近优范围</th></tr></thead><tbody>';
                Object.entries(optimization.solution || {}).forEach(([name, value]) => {
                    const range = (optimization.near_optimal_ranges || {})[name] || [null, null];
                    html += `<tr><td>${escapeHtml(name)}</td><td>${formatResearchValue(value)}</td><td>[${range.map(formatResearchValue).join(', ')}]</td></tr>`;
                });
                html += '</tbody></table></div>';
                const robust = optimization.robust_feedback || {};
                if (robust.attempted) {
                    html += `<p class="hint"><strong>结果反馈的稳健候选：</strong>${escapeHtml(JSON.stringify(robust.candidate_solution || {}))}<br>最坏归一化遗憾：${formatResearchValue(robust.nominal_worst_normalized_regret)} → ${formatResearchValue(robust.candidate_worst_normalized_regret)}；改善 ${formatResearchValue((robust.relative_regret_reduction || 0) * 100)}%。由于5%扰动范围尚未由题目确认，系统不会自动替换名义最优解。</p>`;
                }
            } else {
                html += `<p class="research-warning">未得到可行有限最优解：${escapeHtml(optimization.message || '-')}</p>`;
            }
            html += `<p><strong>${escapeHtml(audit.decision || '')}</strong></p><p class="hint">${escapeHtml(optimization.note || '')}</p>`;
        }
        const graph = specialized.graph_network;
        if (graph) {
            html += `<h4>网络结构 · ${escapeHtml(graph.dataset)}</h4><div class="research-metrics"><span><small>节点</small><strong>${graph.n_nodes}</strong></span><span><small>唯一边</small><strong>${graph.n_unique_edges}</strong></span><span><small>连通分量</small><strong>${graph.connected_components}</strong></span><span><small>网络密度</small><strong>${formatResearchValue(graph.density)}</strong></span></div><p class="hint">${escapeHtml(graph.note || '')}</p>`;
        }
        const simulation = specialized.simulation;
        if (simulation) {
            html += `<h4>Bootstrap 不确定性 · ${escapeHtml(simulation.dataset)}.${escapeHtml(simulation.variable)}</h4><div class="research-metrics"><span><small>观测均值</small><strong>${formatResearchValue(simulation.observed_mean)}</strong></span><span><small>标准差</small><strong>${formatResearchValue(simulation.observed_std)}</strong></span><span><small>均值95%区间</small><strong>[${simulation.mean_confidence_interval_95.map(formatResearchValue).join(', ')}]</strong></span><span><small>仿真次数</small><strong>${simulation.iterations}</strong></span></div><p class="hint">${escapeHtml(simulation.note || '')}</p>`;
        }
        const dynamics = specialized.time_dynamics;
        if (dynamics) {
            html += `<h4>时序动力特征 · ${escapeHtml(dynamics.dataset)}.${escapeHtml(dynamics.variable)}</h4><div class="research-metrics"><span><small>时间点</small><strong>${dynamics.n_time_points}</strong></span><span><small>日趋势</small><strong>${formatResearchValue(dynamics.linear_trend_per_day)}</strong></span><span><small>残差标准差</small><strong>${formatResearchValue(dynamics.residual_std)}</strong></span><span><small>中位间隔/天</small><strong>${formatResearchValue(dynamics.median_interval_days)}</strong></span></div><p class="hint">${escapeHtml(dynamics.note || '')}</p>`;
        }
        const equation = specialized.equation_discovery;
        if (equation) {
            const audit = equation.credibility_audit || {};
            const auditClass = audit.status === 'pass' ? 'research-safe' : (audit.status === 'fail' ? 'research-fail' : 'research-risk');
            html += `<h4>积分弱形式候选方程 · ${escapeHtml(equation.dataset)}.${escapeHtml(equation.target)} <span class="${auditClass}">${escapeHtml(audit.label || '-')}</span></h4>`;
            html += `<pre class="code-block">${escapeHtml(equation.equation || '-')}</pre><div class="research-metrics"><span><small>时间点</small><strong>${equation.n_time_points}</strong></span><span><small>训练窗口</small><strong>${equation.training_windows}</strong></span><span><small>验证窗口</small><strong>${equation.validation_windows}</strong></span><span><small>验证 R²</small><strong>${formatResearchValue((equation.metrics || {}).validation_r2)}</strong></span><span><small>项集稳定性</small><strong>${formatResearchValue((equation.metrics || {}).support_jaccard)}</strong></span></div><p><strong>${escapeHtml(audit.decision || '')}</strong></p><p class="hint">${escapeHtml(equation.note || '')}</p>`;
        }
        const causal = specialized.causal_effect;
        if (causal) {
            const audit = causal.credibility_audit || {};
            const auditClass = audit.status === 'pass' ? 'research-safe' : (audit.status === 'fail' ? 'research-fail' : 'research-risk');
            html += `<h4>正交化因果效应 · ${escapeHtml(causal.treatment)} → ${escapeHtml(causal.outcome)} <span class="${auditClass}">${escapeHtml(audit.label || '-')}</span></h4>`;
            html += `<div class="research-metrics"><span><small>处理效应</small><strong>${formatResearchValue(causal.effect)}</strong></span><span><small>95%区间</small><strong>[${(causal.confidence_interval_95 || []).map(formatResearchValue).join(', ')}]</strong></span><span><small>p值</small><strong>${formatResearchValue(causal.p_value)}</strong></span><span><small>安慰剂p值</small><strong>${formatResearchValue(causal.placebo_p_value)}</strong></span><span><small>重叠/残差变异</small><strong>${formatResearchValue((causal.overlap_share || 0) * 100)}%</strong></span></div><p><strong>${escapeHtml(audit.decision || '')}</strong></p><p class="hint">${escapeHtml(causal.note || '')}</p>`;
        }
        const structures = specialized.data_structure || [];
        structures.forEach(structure => {
            const audit = structure.credibility_audit || {};
            const auditClass = audit.status === 'pass' ? 'research-safe' : (audit.status === 'fail' ? 'research-fail' : 'research-risk');
            html += `<h4>潜在结构与稳健异常 · ${escapeHtml(structure.dataset)} <span class="${auditClass}">${escapeHtml(audit.label || '未审计')}</span></h4>`;
            html += `<div class="research-metrics"><span><small>原始维数</small><strong>${structure.original_dimensions}</strong></span><span><small>90%解释率维数</small><strong>${structure.dimensions_90}</strong></span><span><small>累计解释率</small><strong>${formatResearchValue(structure.cumulative_explained_variance * 100)}%</strong></span><span><small>结构异常</small><strong>${structure.anomaly_count} / ${structure.analysis_rows}</strong></span><span><small>扰动名单一致率</small><strong>${formatResearchValue(structure.anomaly_perturbation_jaccard * 100)}%</strong></span><span><small>子空间稳定性</small><strong>${formatResearchValue(structure.subspace_stability)}</strong></span></div>`;
            html += `<p><strong>${escapeHtml(audit.decision || '')}</strong></p><p class="hint">${escapeHtml(structure.note || '')}</p>`;
            if ((structure.top_anomalies || []).length) {
                html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>原始行索引</th><th>稳健异常分数</th><th>重构误差</th><th>主要偏离变量</th><th>是否越阈值</th></tr></thead><tbody>';
                structure.top_anomalies.slice(0, 10).forEach(row => {
                    const deviations = (row.dominant_deviations || []).map(item => item.feature).join('、') || '-';
                    html += `<tr><td>${escapeHtml(row.row_index)}</td><td>${formatResearchValue(row.robust_z)}</td><td>${formatResearchValue(row.reconstruction_error)}</td><td>${escapeHtml(deviations)}</td><td>${row.flagged ? '<span class="research-fail">是</span>' : '否'}</td></tr>`;
                });
                html += '</tbody></table></div>';
            }
        });
        const customResults = specialized.custom || {};
        Object.entries(customResults).forEach(([taskType, payload]) => {
            html += `<h4>扩展分析器 · ${escapeHtml(taskType)}</h4><pre class="code-block">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>`;
        });
        html += '</details>';
    }

    const modelResults = (result.model_results || []).length ? result.model_results : (result.model_result ? [result.model_result] : []);
    modelResults.forEach((model, modelIndex) => {
        const modelSubject = model.target ? `${model.dataset}.${model.target}` : model.dataset;
        const modelTitle = model.task_type === 'clustering' ? '自动聚类' : '自动预测模型';
        const sequence = modelResults.length > 1 ? ` ${modelIndex + 1}/${modelResults.length}` : '';
        html += `<details class="research-section" open><summary>${modelTitle}${sequence}：${escapeHtml(modelSubject)}</summary>`;
        html += `<p>任务：${escapeHtml(model.task_type)}　最佳模型：<strong>${escapeHtml(model.best_model || '-')}</strong>${model.best_k ? '　簇数：' + model.best_k : ''}　样本/特征：${model.n_samples}/${model.n_features}</p><div class="research-metrics">`;
        Object.entries(model.metrics || {}).forEach(([key, value]) => { html += `<span><small>${escapeHtml(key)}</small><strong>${formatResearchValue(value)}</strong></span>`; });
        html += '</div>';
        const feedback = model.feedback_optimization || {};
        if (feedback.enabled !== undefined) {
            const feedbackStatus = feedback.accepted ? '<span class="research-safe">已采用调优结果</span>' : (feedback.attempted ? '<span class="research-risk">保留基线</span>' : '<span>未执行</span>');
            html += `<h4 style="margin:12px 0 6px;">验证结果反馈优化 ${feedbackStatus}</h4><p class="hint">${escapeHtml(feedback.reason || '')}</p>`;
            if (feedback.attempted) {
                html += `<div class="research-metrics"><span><small>基线 ${escapeHtml(feedback.primary_metric || '')}</small><strong>${formatResearchValue(feedback.baseline_score)}</strong></span><span><small>调优结果</small><strong>${formatResearchValue(feedback.tuned_score)}</strong></span><span><small>相对改善</small><strong>${formatResearchValue((feedback.relative_gain || 0) * 100)}%</strong></span><span><small>改善概率</small><strong>${formatResearchValue((feedback.improvement_probability || 0) * 100)}%</strong></span><span><small>采用阈值</small><strong>${formatResearchValue((feedback.acceptance_threshold || 0) * 100)}%</strong></span><span><small>独立确认样本</small><strong>${feedback.confirmation_samples || 0}</strong></span></div>`;
                html += `<p class="hint">候选依据：${escapeHtml(feedback.candidate_selection_reason || '-')}<br>确认方式：${escapeHtml(feedback.confirmation || '-')}；参数选择与最终复核使用不同数据。</p>`;
            }
            const recommendations = ((feedback.diagnostics || {}).recommendations || []);
            if (recommendations.length) html += `<p class="hint">诊断：${escapeHtml(recommendations.join('；'))}</p>`;
        }
        const credibility = model.credibility_audit || {};
        if (credibility.enabled !== undefined) {
            const auditClass = credibility.status === 'pass' ? 'research-safe' : (credibility.status === 'fail' ? 'research-fail' : 'research-risk');
            html += `<h4 style="margin:12px 0 6px;">结果可信度审计 <span class="${auditClass}">${escapeHtml(credibility.label || '证据不足')}</span></h4>`;
            html += `<p><strong>${escapeHtml(credibility.decision || '')}</strong></p><p class="hint">${escapeHtml(credibility.summary || '')}</p>`;
            if ((credibility.checks || []).length) {
                html += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>审计项</th><th>状态</th><th>证据</th><th>建议</th></tr></thead><tbody>';
                credibility.checks.forEach(check => {
                    const checkClass = check.status === 'pass' ? 'research-safe' : (check.status === 'fail' ? 'research-fail' : 'research-risk');
                    const statusText = check.status === 'pass' ? '通过' : (check.status === 'fail' ? '失败' : (check.status === 'warning' ? '警告' : '未评估'));
                    html += `<tr><td>${escapeHtml(check.name || '-')}</td><td><span class="${checkClass}">${statusText}</span></td><td>${escapeHtml(check.evidence || '-')}</td><td>${escapeHtml(check.recommendation || '-')}</td></tr>`;
                });
                html += '</tbody></table></div>';
            }
            if ((credibility.next_actions || []).length) {
                html += `<p class="research-warning"><strong>优先处理：</strong>${escapeHtml(credibility.next_actions.join('；'))}</p>`;
            }
        }
        const predictionInterval = model.prediction_interval;
        if (predictionInterval) {
            const intervalAudit = predictionInterval.credibility_audit || {};
            const intervalClass = intervalAudit.status === 'pass' ? 'research-safe' : (intervalAudit.status === 'fail' ? 'research-fail' : 'research-risk');
            html += `<h4 style="margin:12px 0 6px;">保序预测区间 <span class="${intervalClass}">${escapeHtml(intervalAudit.label || '-')}</span></h4><div class="research-metrics"><span><small>目标覆盖率</small><strong>${formatResearchValue(predictionInterval.target_coverage * 100)}%</strong></span><span><small>经验覆盖率</small><strong>${predictionInterval.empirical_coverage == null ? '-' : formatResearchValue(predictionInterval.empirical_coverage * 100) + '%'}</strong></span><span><small>平均宽度</small><strong>${formatResearchValue(predictionInterval.mean_interval_width)}</strong></span><span><small>归一化宽度</small><strong>${formatResearchValue(predictionInterval.normalized_width)}</strong></span><span><small>校准样本</small><strong>${predictionInterval.calibration_samples}</strong></span></div><p class="hint">${escapeHtml(predictionInterval.note || '')}</p>`;
        }
        if ((model.feature_join_audit || []).length) {
            html += '<h4 style="margin:12px 0 6px;">跨表特征时间审计</h4><div class="table-wrapper"><table class="data-table"><thead><tr><th>来源</th><th>策略</th><th>特征数</th><th>说明</th></tr></thead><tbody>';
            model.feature_join_audit.forEach(item => { html += `<tr><td>${escapeHtml(item.dataset || '-')}</td><td>${escapeHtml(item.strategy || '-')}</td><td>${item.features_added || 0}</td><td>${escapeHtml(item.reason || '-')}</td></tr>`; });
            html += '</tbody></table></div>';
        }
        if ((model.feature_importance || []).length) {
            html += '<h4 style="margin:12px 0 6px;">跨表建模重要特征</h4><div class="table-wrapper"><table class="data-table"><thead><tr><th>特征</th><th>重要性</th></tr></thead><tbody>';
            model.feature_importance.slice(0, 12).forEach(row => {
                html += `<tr><td>${escapeHtml(row.feature || row.column || '-')}</td><td>${formatResearchValue(row.importance ?? row.score)}</td></tr>`;
            });
            html += '</tbody></table></div>';
        }
        html += `<p class="hint">${escapeHtml(model.note || '')}</p></details>`;
    });

    if (result.ranking_result) {
        const ranking = result.ranking_result;
        html += '<details class="research-section" open><summary>熵权 TOPSIS 综合评价</summary><div class="research-metrics">';
        Object.entries(ranking.weights || {}).slice(0, 10).forEach(([key, value]) => { html += `<span><small>${escapeHtml(key)} 权重</small><strong>${formatResearchValue(value)}</strong></span>`; });
        const rankingAudit = ranking.credibility_audit || {};
        const sensitivity = ranking.sensitivity || {};
        if (rankingAudit.status) {
            const auditClass = rankingAudit.status === 'pass' ? 'research-safe' : (rankingAudit.status === 'fail' ? 'research-fail' : 'research-risk');
            html += `<span><small>排名可信度</small><strong class="${auditClass}">${escapeHtml(rankingAudit.label || '-')}</strong></span><span><small>扰动秩相关</small><strong>${formatResearchValue(sensitivity.median_rank_spearman)}</strong></span><span><small>首名保持率</small><strong>${formatResearchValue((sensitivity.winner_retention || 0) * 100)}%</strong></span>`;
        }
        html += '</div><div class="table-wrapper"><table class="data-table"><thead><tr><th>排名</th><th>对象</th><th>得分</th></tr></thead><tbody>';
        (ranking.ranking || []).slice(0, 15).forEach(row => { html += `<tr><td>${row.rank}</td><td>${escapeHtml(row.entity)}</td><td>${formatResearchValue(row.score)}</td></tr>`; });
        html += '</tbody></table></div>';
        const pareto = ranking.pareto_analysis || {};
        if (pareto.front_size !== undefined) {
            html += `<h4>无权重 Pareto 非支配集</h4><div class="research-metrics"><span><small>审计样本</small><strong>${pareto.sample_size}</strong></span><span><small>非支配方案</small><strong>${pareto.front_size}</strong></span><span><small>非支配比例</small><strong>${formatResearchValue((pareto.front_share || 0) * 100)}%</strong></span><span><small>冲突指标对</small><strong>${(pareto.conflicting_indicator_pairs || []).length}</strong></span></div><p class="hint">${escapeHtml(pareto.note || '')}</p>`;
        }
        html += `<p class="hint">${escapeHtml(rankingAudit.decision || ranking.note || '')}</p></details>`;
    }

    if ((result.charts || []).length) {
        html += '<details class="research-section" open><summary>自动生成图表</summary><div class="research-chart-grid">';
        result.charts.forEach(chart => {
            // The visible caption already names the chart. An empty alt marks the
            // image as decorative and prevents text exports from repeating every title.
            html += `<figure><img src="${chart.url}" alt="" loading="lazy"><figcaption>${escapeHtml(chart.title)}</figcaption></figure>`;
        });
        html += '</div></details>';
    }

    html += '<details class="research-section" open><summary>结论与限制</summary><ul class="research-findings">';
    (result.conclusions || []).forEach(text => { html += `<li>${escapeHtml(text)}</li>`; });
    (result.warnings || []).forEach(text => { html += `<li class="research-warning">${escapeHtml(text)}</li>`; });
    html += '</ul></details>';
    html += `<div class="research-actions"><a class="btn btn-primary" href="${result.evidence_url || '/api/research/evidence'}">📥 下载机器可读证据</a><a class="btn btn-secondary" href="${result.report_url}">下载论证摘要</a><a class="btn btn-secondary" href="${result.manifest_url || '/api/research/manifest'}">下载产物清单</a><button class="btn btn-secondary" onclick="clearResearchCache(this)">清理本次缓存</button><button class="btn btn-secondary" onclick="goStep(4)">继续调整模型</button></div>`;
    box.innerHTML = html;
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
            uploadedFiles.forEach((file, index) => {
                file.is_active = index === Number(data.active_index ?? 0);
            });
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
            selectedSheets.clear();
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
            selectedSheets.clear();
            uploadedFiles.forEach((file, fileIndex) => {
                file.is_active = fileIndex === Number(data.active_index);
            });
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

    let rowCount = 0;
    let rows = '';
    uploadedFiles.forEach((file, fIdx) => {
        const sheets = file.sheets && file.sheets.length ? file.sheets : [null];
        const icon = file.ext && file.ext.includes('xls') ? '📊' : '📄';
        sheets.forEach((sheet, sIdx) => {
            const key = sheetSelectionKey(fIdx, sheet);
            const checked = selectedSheets.has(key);
            const sheetActive = Boolean(
                file.is_active && (file.active_sheet || null) === sheet
            );
            rowCount += 1;
            rows += `
                <tr class="merge-sheet-row ${sheetActive ? 'active' : ''} ${checked ? 'selected' : ''}">
                    <td class="merge-sheet-check-cell"><input type="checkbox" class="merge-sheet-checkbox" data-file-index="${fIdx}" data-sheet-index="${sIdx}" ${checked ? 'checked' : ''} onchange="toggleSheetSelection(this)" aria-label="选择 ${escapeHtml(file.filename)} ${escapeHtml(sheet || '默认表')}"></td>
                    <td><strong>${icon} ${escapeHtml(file.filename)}</strong></td>
                    <td>${escapeHtml(sheet || '默认表')}</td>
                    <td>${Number(file.shape?.[0] || 0).toLocaleString()} × ${Number(file.shape?.[1] || 0).toLocaleString()}</td>
                    <td>${sheetActive ? '<span class="merge-current-badge">● 当前</span>' : '-'}</td>
                    <td class="merge-sheet-actions"><button class="btn btn-sm" onclick="selectSheetByIndex(${fIdx}, ${sIdx})">查看</button>${sIdx === 0 ? `<button class="btn btn-sm merge-delete-btn" onclick="deleteDataset(${fIdx})" title="删除整个文件">🗑️</button>` : ''}</td>
                </tr>`;
        });
    });

    listEl.innerHTML = rowCount ? `
        <div class="merge-selection-toolbar">
            <span>共 ${rowCount} 个可选表</span>
            <span id="merge-selection-summary">已选择 0 个</span>
        </div>
        <div class="table-wrapper merge-sheet-table-wrapper">
            <table class="data-table merge-sheet-table">
                <thead><tr><th><input type="checkbox" id="merge-select-all" onchange="toggleAllMergeSheets(this)" aria-label="全选可合并表"></th><th>文件</th><th>Sheet</th><th>规模</th><th>状态</th><th>操作</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>` : '<p class="hint">尚未上传可用数据表。</p>';
    updateMergeSelectionState();
    updateJoinOptions();
}

function sheetSelectionKey(fileIndex, sheetName) {
    return JSON.stringify([Number(fileIndex), sheetName ?? null]);
}

function toggleSheetSelection(checkbox) {
    const fileIndex = Number(checkbox.dataset.fileIndex);
    const sheetIndex = Number(checkbox.dataset.sheetIndex);
    const file = uploadedFiles[fileIndex];
    const sheets = file && file.sheets && file.sheets.length ? file.sheets : [null];
    if (!file || !Number.isInteger(sheetIndex) || sheetIndex < 0 || sheetIndex >= sheets.length) return;
    const key = sheetSelectionKey(fileIndex, sheets[sheetIndex]);
    if (checkbox.checked) {
        selectedSheets.add(key);
    } else {
        selectedSheets.delete(key);
    }
    checkbox.closest('tr')?.classList.toggle('selected', checkbox.checked);
    updateMergeSelectionState();
}

function toggleAllMergeSheets(master) {
    document.querySelectorAll('.merge-sheet-checkbox').forEach(checkbox => {
        checkbox.checked = master.checked;
        toggleSheetSelection(checkbox);
    });
    updateMergeSelectionState();
}

function updateMergeSelectionState() {
    const checkboxes = Array.from(document.querySelectorAll('.merge-sheet-checkbox'));
    const selectedCount = checkboxes.filter(item => item.checked).length;
    const selectAll = document.getElementById('merge-select-all');
    if (selectAll) {
        selectAll.checked = checkboxes.length > 0 && selectedCount === checkboxes.length;
        selectAll.indeterminate = selectedCount > 0 && selectedCount < checkboxes.length;
    }
    const summary = document.getElementById('merge-selection-summary');
    if (summary) summary.textContent = `已选择 ${selectedCount} 个`;
    const button = document.getElementById('execute-merge-btn');
    if (button) {
        button.disabled = selectedCount < 2;
        button.textContent = selectedCount >= 2 ? `执行合并（${selectedCount} 个表）` : '执行合并';
    }
}

function selectSheetByIndex(fileIndex, sheetIndex) {
    const file = uploadedFiles[fileIndex];
    const sheets = file && file.sheets && file.sheets.length ? file.sheets : [null];
    if (!file || sheetIndex < 0 || sheetIndex >= sheets.length) return;
    selectSheet(fileIndex, sheets[sheetIndex]);
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
            if (currentStep === 2) await loadEDA();
            return true;
        } else {
            showToast(data.error || '切换失败', 'error');
            return false;
        }
    } catch (e) {
        showToast('切换出错: ' + e.message, 'error');
        return false;
    }
}

function onMultiOpChange() {
    const type = document.getElementById('multi-op-type').value;
    document.getElementById('merge-options').classList.toggle('hidden', type !== 'merge');
    document.getElementById('join-options').classList.toggle('hidden', type !== 'join');
    document.getElementById('transform-options').classList.toggle('hidden', type !== 'transform');
    if (type === 'merge') updateMergeSelectionState();
    if (type === 'transform') loadTransformCapabilities();
}

async function executeMerge() {
    if (selectedSheets.size < 2) {
        showToast('请至少勾选2个sheet进行合并', 'error');
        return;
    }
    const axis = parseInt(document.getElementById('merge-axis').value);
    const sources = Array.from(selectedSheets).map(key => {
        const [fileIndex, sheetName] = JSON.parse(key);
        return { file_index: Number(fileIndex), sheet_name: sheetName };
    }).filter(source => {
        const file = uploadedFiles[source.file_index];
        const sheets = file && file.sheets && file.sheets.length ? file.sheets : [null];
        return Boolean(file && sheets.includes(source.sheet_name));
    });
    if (sources.length < 2) {
        selectedSheets.clear();
        renderMultiTablePanel();
        showToast('选择状态已过期，请重新勾选至少2个表', 'error');
        return;
    }
    
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
            const diagnostics = data.merge_diagnostics || {};
            const schemaWarning = axis === 0 && diagnostics.schemas_identical === false
                ? `；列结构不同，已按 ${diagnostics.union_columns?.length || data.shape[1]} 列并集对齐，缺失位置留空`
                : '';
            document.getElementById('data-status').textContent = `合并结果: ${data.shape[0]}行×${data.shape[1]}列${schemaWarning}`;
            populateTargetOptions(data.data.columns, data.target_hint);
            showToast(`合并成功！${data.shape[0]}行×${data.shape[1]}列${schemaWarning}`);
            goStep(2);
        } else {
            showToast(data.error || '合并失败', 'error');
        }
    } catch (e) {
        showToast('合并出错: ' + e.message, 'error');
    }
}

async function loadTransformCapabilities() {
    const select = document.getElementById('transform-operation');
    if (!select) return;
    try {
        const response = await fetch('/api/data/transform/capabilities');
        const data = await response.json();
        if (!data.success) throw new Error(data.error || '操作注册表加载失败');
        transformCapabilities = data.capabilities || [];
        const categories = new Map();
        transformCapabilities.forEach(item => {
            if (!categories.has(item.category)) categories.set(item.category, []);
            categories.get(item.category).push(item);
        });
        let html = '<option value="">选择一个操作模板...</option>';
        categories.forEach((items, category) => {
            html += `<optgroup label="${escapeHtml(category)}">`;
            items.forEach(item => {
                const unavailable = item.availability === 'unavailable';
                const suffix = unavailable ? ' · 当前不可用' : (item.availability === 'review' ? ' · 需核验' : ' · 可直接配置');
                html += `<option value="${escapeHtml(item.name)}" ${unavailable ? 'disabled' : ''} title="${escapeHtml(item.availability_reason || '')}">${escapeHtml(item.label)}${suffix}</option>`;
            });
            html += '</optgroup>';
        });
        select.innerHTML = html;
        const ready = transformCapabilities.filter(item => item.availability === 'ready').length;
        const review = transformCapabilities.filter(item => item.availability === 'review').length;
        const unavailable = transformCapabilities.filter(item => item.availability === 'unavailable').length;
        const summary = document.getElementById('transform-capability-summary');
        if (summary) {
            summary.innerHTML = `<span class="is-ready">${ready} 个已绑定</span><span class="is-review">${review} 个需核验</span><span class="is-unavailable">${unavailable} 个不适用</span><small>不可用操作不会再写入伪造字段。</small>`;
        }
        renderTransformQuickActions();
        syncTransformPipelineCards();
    } catch (error) {
        select.innerHTML = '<option value="">操作注册表加载失败</option>';
        console.error('加载表变换注册表失败', error);
    }
}

function setTransformGoalExample(text) {
    const input = document.getElementById('transform-goal');
    if (!input) return;
    input.value = text;
    input.focus();
}

function renderTransformQuickActions() {
    const box = document.getElementById('transform-quick-actions');
    if (!box) return;
    const preferred = [
        ['fill_missing', '处理缺失'],
        ['deduplicate', '删除重复'],
        ['aggregate', '分组汇总'],
        ['time_features', '时间特征'],
        ['window_features', '滚动指标'],
        ['normalize', '标准化'],
    ];
    const actions = preferred.map(([name, simpleLabel]) => ({
        capability: transformCapabilities.find(item => item.name === name),
        simpleLabel,
    })).filter(item => item.capability && item.capability.availability !== 'unavailable');
    if (!actions.length) {
        box.innerHTML = '<span class="hint">当前表没有可直接添加的常用动作，请展开“添加其他操作”查看原因。</span>';
        return;
    }
    box.innerHTML = actions.map(({ capability, simpleLabel }) => `
        <button type="button" onclick="addTransformOperationByName('${escapeHtml(capability.name)}')" title="${escapeHtml(capability.availability_reason || capability.description || '')}">
            <span>＋</span><strong>${escapeHtml(simpleLabel)}</strong>${capability.availability === 'review' ? '<small>需确认</small>' : ''}
        </button>`).join('');
}

function addTransformOperationByName(name) {
    const select = document.getElementById('transform-operation');
    if (!select) return;
    select.value = name;
    insertTransformTemplate();
}

function readTransformPipeline() {
    const editor = document.getElementById('transform-pipeline');
    let pipeline;
    try {
        pipeline = JSON.parse(editor.value.trim());
    } catch (error) {
        throw new Error('流水线不是合法 JSON：' + error.message);
    }
    if (!Array.isArray(pipeline) || pipeline.length === 0) {
        throw new Error('流水线必须是至少包含一个步骤的 JSON 数组');
    }
    return pipeline;
}

function insertTransformTemplate() {
    const select = document.getElementById('transform-operation');
    const capability = transformCapabilities.find(item => item.name === select.value);
    if (!capability) return;
    if (capability.availability === 'unavailable') {
        const reason = capability.availability_reason || '当前数据结构不满足该操作的输入要求';
        document.getElementById('transform-status').textContent = `“${capability.label}”当前不可用：${reason}`;
        showToast(reason, 'error');
        select.value = '';
        return;
    }
    const editor = document.getElementById('transform-pipeline');
    let pipeline = [];
    if (editor.value.trim()) {
        try {
            pipeline = JSON.parse(editor.value);
            if (!Array.isArray(pipeline)) throw new Error('不是数组');
        } catch (error) {
            showToast('请先修正当前流水线 JSON，再添加模板', 'error');
            select.value = '';
            return;
        }
    }
    pipeline.push({ operation: capability.name, params: capability.template });
    transformEditingStepIndex = pipeline.length - 1;
    updateTransformPipeline(pipeline);
    const review = capability.availability === 'review' ? `；仍需核验：${capability.availability_reason}` : '';
    document.getElementById('transform-status').textContent = `已添加“${capability.label}”，字段已绑定当前表${review}。`;
    select.value = '';
}

function transformParamSummary(params) {
    const labels = {
        group_by: '分组', aggregations: '聚合', columns: '字段', column: '字段',
        expression: '公式', output: '输出', how: '方式', on: '键', left_on: '左键',
        right_on: '右键', periods: '窗口', date_column: '时间', value_column: '数值',
    };
    return Object.entries(params || {}).slice(0, 5).map(([key, value]) => {
        let rendered;
        if (Array.isArray(value)) {
            rendered = value.map(item => {
                if (item && typeof item === 'object') {
                    return interactiveVizDisplayField(item.output)
                        || `${interactiveVizDisplayField(item.column)} ${item.function || ''}`.trim()
                        || '配置项';
                }
                return String(item);
            }).join('、');
        } else if (value && typeof value === 'object') {
            rendered = Object.keys(value).join('、');
        } else {
            rendered = String(value ?? '-');
        }
        if (rendered.length > 52) rendered = rendered.slice(0, 50) + '…';
        return `<span><small>${escapeHtml(labels[key] || key)}</small>${escapeHtml(rendered || '-')}</span>`;
    }).join('');
}

function transformStepColumns(step = {}) {
    const columns = [];
    const add = value => {
        if (value === undefined || value === null || value === '' || value === '*') return;
        const name = String(value);
        if (!columns.includes(name)) columns.push(name);
    };
    (uploadedData?.columns || []).forEach(add);
    const params = step.params || {};
    ['columns', 'group_by', 'subset', 'by', 'partition_by', 'value_columns', 'index', 'values', 'id_vars', 'value_vars'].forEach(key => {
        const values = Array.isArray(params[key]) ? params[key] : [];
        values.forEach(add);
    });
    ['column', 'time_column', 'order_by'].forEach(key => add(params[key]));
    (params.aggregations || []).forEach(item => add(item?.column));
    (params.conditions || []).forEach(item => add(item?.column));
    return columns;
}

function transformNumericColumns(step = {}) {
    const columns = transformStepColumns(step);
    const dtypes = uploadedData?.dtypes || {};
    const types = uploadedData?.column_types || {};
    const numeric = columns.filter(column => {
        const dtype = String(dtypes[column] || '').toLowerCase();
        return types[column] === 'numeric' || /int|float|double|decimal|number/.test(dtype);
    });
    const referenced = [
        ...(step.params?.value_columns || []),
        ...(step.params?.aggregations || []).map(item => item?.column),
    ].filter(Boolean).map(String);
    referenced.forEach(column => {
        if (columns.includes(column) && !numeric.includes(column)) numeric.push(column);
    });
    return numeric.length ? numeric : columns;
}

function transformOptionTags(columns, selected, attributes = '') {
    const active = new Set((selected || []).map(String));
    return columns.map(column => `
        <label class="transform-option-chip">
            <input type="checkbox" data-transform-field="${escapeHtml(column)}" ${active.has(String(column)) ? 'checked' : ''} ${attributes}>
            <span>${escapeHtml(column)}</span>
        </label>`).join('');
}

function transformSelectOptions(columns, selected) {
    return columns.map(column => `<option value="${escapeHtml(column)}" ${String(column) === String(selected) ? 'selected' : ''}>${escapeHtml(column)}</option>`).join('');
}

function renderAggregateStepEditor(index, step, columns, numeric) {
    const params = step.params || {};
    const groupBy = params.group_by || [];
    const specs = params.aggregations || [];
    const specByColumn = new Map(specs.filter(item => item?.column && item.column !== '*').map(item => [String(item.column), item]));
    const metricColumns = Array.from(new Set([...numeric, ...specByColumn.keys()]));
    const functions = [['sum', '合计'], ['mean', '平均'], ['median', '中位数'], ['min', '最小'], ['max', '最大'], ['count', '非空计数']];
    return `
        <div class="transform-editor-section">
            <strong>按什么分组</strong><small>可多选；不选表示整表汇总</small>
            <div class="transform-editor-options" data-editor-role="aggregate-groups">${transformOptionTags(columns, groupBy, `onchange="applyAggregateStepEditor(${index})"`)}</div>
        </div>
        <div class="transform-editor-section">
            <strong>汇总哪些指标</strong><small>至少选择一个，右侧选择算法</small>
            <div class="transform-metric-list" data-editor-role="aggregate-metrics">
                ${metricColumns.map(column => {
                    const spec = specByColumn.get(String(column));
                    const fn = spec?.function || 'sum';
                    return `<div class="transform-metric-row">
                        <label><input type="checkbox" data-transform-field="${escapeHtml(column)}" ${spec ? 'checked' : ''} onchange="applyAggregateStepEditor(${index}, this)"><span>${escapeHtml(column)}</span></label>
                        <select aria-label="${escapeHtml(column)}的汇总方式" onchange="applyAggregateStepEditor(${index})">${functions.map(([value, label]) => `<option value="${value}" ${value === fn ? 'selected' : ''}>${label}</option>`).join('')}</select>
                    </div>`;
                }).join('')}
            </div>
        </div>`;
}

function renderFillMissingStepEditor(index, step, columns, numeric) {
    const params = step.params || {};
    const strategy = String(params.strategy || 'median');
    const numericOnly = ['mean', 'median', 'interpolate'].includes(strategy);
    const eligible = numericOnly ? numeric : columns;
    const selected = (params.columns || []).filter(column => eligible.includes(String(column)));
    const strategies = [['median', '中位数（数值）'], ['mean', '平均数（数值）'], ['mode', '众数'], ['forward', '向前填充'], ['backward', '向后填充'], ['interpolate', '线性插值（数值）'], ['constant', '固定值'], ['drop_rows', '删除缺失行']];
    return `
        <div class="transform-editor-section">
            <strong>处理字段</strong><small>${numericOnly ? '当前策略只支持数值字段' : '可多选'}</small>
            <div class="transform-editor-options" data-editor-role="fill-columns">${transformOptionTags(eligible, selected, `onchange="applyFillMissingStepEditor(${index}, this)"`)}</div>
        </div>
        <div class="transform-editor-section transform-editor-inline">
            <label><span>处理方式</span><select data-editor-role="fill-strategy" onchange="changeFillMissingStrategy(${index}, this.value)">${strategies.map(([value, label]) => `<option value="${value}" ${value === strategy ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
            ${strategy === 'constant' ? `<label><span>填充值</span><input data-editor-role="fill-value" value="${escapeHtml(params.value ?? '')}" onchange="applyFillMissingStepEditor(${index})" placeholder="例如 0 或 未知"></label>` : ''}
        </div>
        <div class="transform-editor-section">
            <strong>组内填补（可选）</strong><small>例如每个地区分别计算中位数</small>
            <div class="transform-editor-options" data-editor-role="fill-groups">${transformOptionTags(columns, params.group_by || [], `onchange="applyFillMissingStepEditor(${index})"`)}</div>
        </div>`;
}

function renderDeduplicateStepEditor(index, step, columns) {
    const params = step.params || {};
    const keep = params.keep === false ? 'false' : String(params.keep || 'first');
    return `
        <div class="transform-editor-section">
            <strong>判断重复的字段</strong><small>不选择时按整行判断</small>
            <div class="transform-editor-options" data-editor-role="deduplicate-columns">${transformOptionTags(columns, params.subset || [], `onchange="applyDeduplicateStepEditor(${index})"`)}</div>
        </div>
        <div class="transform-editor-section transform-editor-inline"><label><span>重复时保留</span><select data-editor-role="deduplicate-keep" onchange="applyDeduplicateStepEditor(${index})"><option value="first" ${keep === 'first' ? 'selected' : ''}>第一条</option><option value="last" ${keep === 'last' ? 'selected' : ''}>最后一条</option><option value="false" ${keep === 'false' ? 'selected' : ''}>重复项全部删除</option></select></label></div>`;
}

function renderSortStepEditor(index, step, columns) {
    const params = step.params || {};
    const selected = (params.by || []).map(String);
    const ascending = Array.isArray(params.ascending) ? params.ascending : selected.map(() => params.ascending !== false);
    return `<div class="transform-editor-section">
        <strong>排序字段</strong><small>按选中顺序依次排序；至少选择一个</small>
        <div class="transform-metric-list" data-editor-role="sort-fields">${columns.map(column => {
            const position = selected.indexOf(String(column));
            return `<div class="transform-metric-row">
                <label><input type="checkbox" data-transform-field="${escapeHtml(column)}" ${position >= 0 ? 'checked' : ''} onchange="applySortStepEditor(${index}, this)"><span>${escapeHtml(column)}</span></label>
                <select onchange="applySortStepEditor(${index})"><option value="true" ${position < 0 || ascending[position] !== false ? 'selected' : ''}>升序</option><option value="false" ${position >= 0 && ascending[position] === false ? 'selected' : ''}>降序</option></select>
            </div>`;
        }).join('')}</div>
    </div>`;
}

function renderColumnSelectionStepEditor(index, step, columns) {
    const params = step.params || {};
    const isSelect = step.operation === 'select_columns';
    return `<div class="transform-editor-section">
        <strong>${isSelect ? '保留这些字段' : '删除这些字段'}</strong><small>${isSelect ? '至少选择一个' : '可多选'}</small>
        <div class="transform-editor-options" data-editor-role="column-selection">${transformOptionTags(columns, params.columns || [], `onchange="applyColumnSelectionStepEditor(${index}, this)"`)}</div>
    </div>`;
}

function renderTimeFeaturesStepEditor(index, step, columns) {
    const params = step.params || {};
    const features = [['year', '年'], ['quarter', '季度'], ['month', '月'], ['week', '周'], ['day', '日'], ['dayofweek', '星期'], ['hour', '小时'], ['is_weekend', '是否周末'], ['month_sin', '月份周期正弦'], ['month_cos', '月份周期余弦']];
    return `
        <div class="transform-editor-section transform-editor-inline"><label><span>时间字段</span><select data-editor-role="time-column" onchange="applyTimeFeaturesStepEditor(${index})">${transformSelectOptions(columns, params.time_column)}</select></label></div>
        <div class="transform-editor-section"><strong>生成特征</strong><small>至少选择一个</small><div class="transform-editor-options" data-editor-role="time-features">${features.map(([value, label]) => `<label class="transform-option-chip"><input type="checkbox" data-transform-feature="${value}" ${(params.features || []).includes(value) ? 'checked' : ''} onchange="applyTimeFeaturesStepEditor(${index}, this)"><span>${label}</span></label>`).join('')}</div></div>`;
}

function renderWindowStepEditor(index, step, columns, numeric) {
    const params = step.params || {};
    const specs = params.features || [];
    const kinds = new Set(specs.map(item => String(item.kind || '')));
    const periods = Number(specs.find(item => ['lag', 'diff', 'pct_change'].includes(item.kind))?.periods || 1);
    const windowSize = Number(specs.find(item => String(item.kind || '').startsWith('rolling_'))?.window || 7);
    return `
        <div class="transform-editor-section transform-editor-inline"><label><span>排序/时间字段</span><select data-editor-role="window-order" onchange="applyWindowStepEditor(${index})">${transformSelectOptions(columns, params.order_by)}</select></label><label><span>滞后期数</span><input type="number" min="1" value="${periods}" data-editor-role="window-periods" onchange="applyWindowStepEditor(${index})"></label><label><span>滚动窗口</span><input type="number" min="1" value="${windowSize}" data-editor-role="window-size" onchange="applyWindowStepEditor(${index})"></label></div>
        <div class="transform-editor-section"><strong>分组实体（可选）</strong><small>例如每个地区分别计算</small><div class="transform-editor-options" data-editor-role="window-groups">${transformOptionTags(columns, params.partition_by || [], `onchange="applyWindowStepEditor(${index})"`)}</div></div>
        <div class="transform-editor-section"><strong>数值指标</strong><small>至少选择一个</small><div class="transform-editor-options" data-editor-role="window-values">${transformOptionTags(numeric, params.value_columns || [], `onchange="applyWindowStepEditor(${index}, this)"`)}</div></div>
        <div class="transform-editor-section"><strong>计算内容</strong><small>可组合</small><div class="transform-editor-options" data-editor-role="window-kinds">${[['lag', '滞后值'], ['diff', '差分'], ['pct_change', '增长率'], ['rolling_mean', '滚动平均'], ['rolling_sum', '滚动合计']].map(([value, label]) => `<label class="transform-option-chip"><input type="checkbox" data-transform-feature="${value}" ${kinds.has(value) ? 'checked' : ''} onchange="applyWindowStepEditor(${index}, this)"><span>${label}</span></label>`).join('')}</div></div>`;
}

function renderNormalizeStepEditor(index, step, numeric) {
    const params = step.params || {};
    const methods = [['zscore', 'Z-score 标准化'], ['minmax', '缩放到 0–1'], ['robust', '稳健缩放'], ['log1p', 'log(1+x) 变换']];
    return `
        <div class="transform-editor-section"><strong>数值字段</strong><small>至少选择一个</small><div class="transform-editor-options" data-editor-role="normalize-columns">${transformOptionTags(numeric, params.columns || [], `onchange="applyNormalizeStepEditor(${index}, this)"`)}</div></div>
        <div class="transform-editor-section transform-editor-inline"><label><span>变换方式</span><select data-editor-role="normalize-method" onchange="applyNormalizeStepEditor(${index})">${methods.map(([value, label]) => `<option value="${value}" ${params.method === value ? 'selected' : ''}>${label}</option>`).join('')}</select></label><label><span>新字段后缀</span><input data-editor-role="normalize-suffix" value="${escapeHtml(params.suffix ?? '_z')}" onchange="applyNormalizeStepEditor(${index})"></label></div>`;
}

function renderFilterStepEditor(index, step, columns) {
    const params = step.params || {};
    const condition = (params.conditions || [])[0] || {};
    const operator = String(condition.operator || 'eq');
    const value = Array.isArray(condition.value) ? condition.value.join(', ') : (condition.value ?? '');
    const operators = [['eq', '等于'], ['ne', '不等于'], ['gt', '大于'], ['ge', '大于等于'], ['lt', '小于'], ['le', '小于等于'], ['contains', '包含文本'], ['not_contains', '不包含文本'], ['in', '属于列表'], ['not_in', '不属于列表'], ['between', '位于区间'], ['is_null', '为空'], ['not_null', '不为空']];
    const noValue = ['is_null', 'not_null'].includes(operator);
    return `<div class="transform-editor-section transform-filter-row" data-editor-role="filter-condition">
        <label><span>字段</span><select data-editor-role="filter-column" onchange="applyFilterStepEditor(${index})">${transformSelectOptions(columns, condition.column)}</select></label>
        <label><span>条件</span><select data-editor-role="filter-operator" onchange="applyFilterStepEditor(${index})">${operators.map(([item, label]) => `<option value="${item}" ${item === operator ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
        <label class="${noValue ? 'is-hidden' : ''}"><span>${['in', 'not_in'].includes(operator) ? '值（逗号分隔）' : operator === 'between' ? '起点, 终点' : '值'}</span><input data-editor-role="filter-value" value="${escapeHtml(value)}" onchange="applyFilterStepEditor(${index})"></label>
    </div>`;
}

function renderTransformStepEditor(index, step) {
    const columns = transformStepColumns(step);
    const numeric = transformNumericColumns(step);
    let body = '';
    if (!columns.length) return '<div class="transform-editor-message">未读取到当前表字段，请重新选择数据集。</div>';
    switch (step.operation) {
        case 'aggregate': body = renderAggregateStepEditor(index, step, columns, numeric); break;
        case 'fill_missing': body = renderFillMissingStepEditor(index, step, columns, numeric); break;
        case 'deduplicate': body = renderDeduplicateStepEditor(index, step, columns); break;
        case 'sort_rows': body = renderSortStepEditor(index, step, columns); break;
        case 'select_columns':
        case 'drop_columns': body = renderColumnSelectionStepEditor(index, step, columns); break;
        case 'time_features': body = renderTimeFeaturesStepEditor(index, step, columns); break;
        case 'window_features': body = renderWindowStepEditor(index, step, columns, numeric); break;
        case 'normalize': body = renderNormalizeStepEditor(index, step, numeric); break;
        case 'filter_rows': body = renderFilterStepEditor(index, step, columns); break;
        default:
            body = '<div class="transform-editor-message"><strong>这个专业操作暂未提供表单编辑器。</strong><span>可在下方“高级模式”中查看参数；执行前仍会做字段和规模校验。</span></div>';
    }
    return `<div class="transform-step-editor" id="transform-step-editor-${index}"><div class="transform-editor-head"><strong>设置这一步</strong><span>修改后立即写入方案，预览结果会自动失效</span></div>${body}</div>`;
}

function transformEditorRoot(index) {
    return document.getElementById(`transform-step-editor-${index}`);
}

function checkedTransformValues(root, selector) {
    return Array.from(root?.querySelectorAll(`${selector}:checked`) || []).map(item => item.dataset.transformField || item.dataset.transformFeature).filter(Boolean);
}

function mutateTransformStep(index, callback, message = '') {
    let pipeline;
    try { pipeline = readTransformPipeline(); } catch (error) { showToast(error.message, 'error'); return false; }
    if (!pipeline[index]) return false;
    callback(pipeline[index].params || (pipeline[index].params = {}), pipeline[index]);
    transformEditingStepIndex = index;
    updateTransformPipeline(pipeline, message);
    return true;
}

function toggleTransformStepEditor(index) {
    transformEditingStepIndex = transformEditingStepIndex === index ? null : index;
    syncTransformPipelineCards();
}

function requireTransformEditorSelection(values, control, message) {
    if (values.length) return true;
    if (control) control.checked = true;
    showToast(message, 'error');
    return false;
}

function applyAggregateStepEditor(index, changedControl = null) {
    const root = transformEditorRoot(index);
    const groups = checkedTransformValues(root, '[data-editor-role="aggregate-groups"] input');
    const rows = Array.from(root?.querySelectorAll('[data-editor-role="aggregate-metrics"] .transform-metric-row') || []);
    const aggregations = rows.filter(row => row.querySelector('input')?.checked).map(row => {
        const column = row.querySelector('input').dataset.transformField;
        const fn = row.querySelector('select').value;
        return { column, function: fn, output: `${column}_${fn}` };
    });
    if (!requireTransformEditorSelection(aggregations, changedControl, '汇总至少需要选择一个指标')) return;
    mutateTransformStep(index, params => { params.group_by = groups; params.aggregations = aggregations; }, '汇总设置已更新，请预览结果。');
}

function changeFillMissingStrategy(index, strategy) {
    let pipeline;
    try { pipeline = readTransformPipeline(); } catch (error) { showToast(error.message, 'error'); return; }
    const step = pipeline[index];
    if (!step) return;
    const params = step.params || (step.params = {});
    params.strategy = strategy;
    if (['mean', 'median', 'interpolate'].includes(strategy)) {
        const numeric = transformNumericColumns(step);
        params.columns = (params.columns || []).filter(column => numeric.includes(String(column)));
        if (!params.columns.length && numeric.length) params.columns = [numeric[0]];
    }
    transformEditingStepIndex = index;
    updateTransformPipeline(pipeline, '缺失值处理方式已更新。');
}

function applyFillMissingStepEditor(index, changedControl = null) {
    const root = transformEditorRoot(index);
    const columns = checkedTransformValues(root, '[data-editor-role="fill-columns"] input');
    if (!requireTransformEditorSelection(columns, changedControl, '至少选择一个需要处理的字段')) return;
    const strategy = root.querySelector('[data-editor-role="fill-strategy"]')?.value || 'median';
    const groups = checkedTransformValues(root, '[data-editor-role="fill-groups"] input');
    const value = root.querySelector('[data-editor-role="fill-value"]')?.value;
    mutateTransformStep(index, params => {
        params.columns = columns; params.strategy = strategy; params.group_by = groups;
        if (strategy === 'constant') params.value = value ?? '';
        else delete params.value;
    }, '缺失值设置已更新，请预览结果。');
}

function applyDeduplicateStepEditor(index) {
    const root = transformEditorRoot(index);
    const subset = checkedTransformValues(root, '[data-editor-role="deduplicate-columns"] input');
    const keepValue = root.querySelector('[data-editor-role="deduplicate-keep"]')?.value || 'first';
    mutateTransformStep(index, params => { params.subset = subset; params.keep = keepValue === 'false' ? false : keepValue; }, '去重设置已更新。');
}

function applySortStepEditor(index, changedControl = null) {
    const root = transformEditorRoot(index);
    const rows = Array.from(root?.querySelectorAll('[data-editor-role="sort-fields"] .transform-metric-row') || []);
    const selected = rows.filter(row => row.querySelector('input')?.checked);
    if (!requireTransformEditorSelection(selected, changedControl, '排序至少需要选择一个字段')) return;
    mutateTransformStep(index, params => {
        params.by = selected.map(row => row.querySelector('input').dataset.transformField);
        params.ascending = selected.map(row => row.querySelector('select').value === 'true');
    }, '排序设置已更新。');
}

function applyColumnSelectionStepEditor(index, changedControl = null) {
    const root = transformEditorRoot(index);
    const columns = checkedTransformValues(root, '[data-editor-role="column-selection"] input');
    let operation = '';
    try { operation = readTransformPipeline()[index]?.operation || ''; } catch (error) { return; }
    if (operation === 'select_columns' && !requireTransformEditorSelection(columns, changedControl, '保留字段至少选择一个')) return;
    mutateTransformStep(index, params => { params.columns = columns; }, '字段选择已更新。');
}

function applyTimeFeaturesStepEditor(index, changedControl = null) {
    const root = transformEditorRoot(index);
    const features = checkedTransformValues(root, '[data-editor-role="time-features"] input');
    if (!requireTransformEditorSelection(features, changedControl, '至少选择一个时间特征')) return;
    const timeColumn = root.querySelector('[data-editor-role="time-column"]')?.value || '';
    mutateTransformStep(index, params => { params.time_column = timeColumn; params.features = features; }, '时间特征设置已更新。');
}

function applyWindowStepEditor(index, changedControl = null) {
    const root = transformEditorRoot(index);
    const values = checkedTransformValues(root, '[data-editor-role="window-values"] input');
    const kinds = checkedTransformValues(root, '[data-editor-role="window-kinds"] input');
    if (!requireTransformEditorSelection(values, changedControl, '至少选择一个数值指标')) return;
    if (!requireTransformEditorSelection(kinds, changedControl, '至少选择一种窗口计算')) return;
    const periods = Math.max(1, Number(root.querySelector('[data-editor-role="window-periods"]')?.value || 1));
    const windowSize = Math.max(1, Number(root.querySelector('[data-editor-role="window-size"]')?.value || 7));
    const features = kinds.map(kind => kind.startsWith('rolling_') ? { kind, window: windowSize, shift: 1 } : { kind, periods });
    mutateTransformStep(index, params => {
        params.order_by = root.querySelector('[data-editor-role="window-order"]')?.value || '';
        params.partition_by = checkedTransformValues(root, '[data-editor-role="window-groups"] input');
        params.value_columns = values;
        params.features = features;
    }, '窗口指标设置已更新。');
}

function applyNormalizeStepEditor(index, changedControl = null) {
    const root = transformEditorRoot(index);
    const columns = checkedTransformValues(root, '[data-editor-role="normalize-columns"] input');
    if (!requireTransformEditorSelection(columns, changedControl, '标准化至少需要选择一个数值字段')) return;
    mutateTransformStep(index, params => {
        params.columns = columns;
        params.method = root.querySelector('[data-editor-role="normalize-method"]')?.value || 'zscore';
        params.suffix = root.querySelector('[data-editor-role="normalize-suffix"]')?.value || '_z';
    }, '标准化设置已更新。');
}

function applyFilterStepEditor(index) {
    const root = transformEditorRoot(index);
    const column = root.querySelector('[data-editor-role="filter-column"]')?.value || '';
    const operator = root.querySelector('[data-editor-role="filter-operator"]')?.value || 'eq';
    const rawValue = root.querySelector('[data-editor-role="filter-value"]')?.value || '';
    let value = rawValue;
    if (['in', 'not_in', 'between'].includes(operator)) value = rawValue.split(',').map(item => item.trim()).filter(Boolean);
    mutateTransformStep(index, params => {
        params.combine = params.combine || 'and';
        params.conditions = [{ column, operator, ...(['is_null', 'not_null'].includes(operator) ? {} : { value }) }];
    }, '筛选条件已更新。');
}

function syncTransformPipelineCards(invalidatePreview = false) {
    const editor = document.getElementById('transform-pipeline');
    const cards = document.getElementById('transform-pipeline-cards');
    const count = document.getElementById('transform-step-count');
    if (!editor || !cards || !count) return;
    if (invalidatePreview) {
        disposeTransformPreviewChart();
        transformPreviewResult = null;
        document.getElementById('transform-result').innerHTML = '';
    }
    const raw = editor.value.trim();
    if (!raw) {
        count.textContent = '0 步';
        cards.innerHTML = '<div class="transform-pipeline-empty">还没有步骤，请生成方案或点击上方常用动作。</div>';
        return;
    }
    let pipeline;
    try {
        pipeline = JSON.parse(raw);
        if (!Array.isArray(pipeline)) throw new Error('根节点不是数组');
    } catch (error) {
        count.textContent = '格式待修正';
        cards.innerHTML = `<div class="transform-pipeline-empty is-error">高级 JSON 暂时无法解析：${escapeHtml(error.message)}</div>`;
        return;
    }
    count.textContent = `${pipeline.length} 步`;
    if (!pipeline.length) {
        cards.innerHTML = '<div class="transform-pipeline-empty">还没有步骤，请生成方案或点击上方常用动作。</div>';
        return;
    }
    cards.innerHTML = pipeline.map((step, index) => {
        const capability = transformCapabilities.find(item => item.name === step.operation);
        const label = capability?.label || step.operation || '未知操作';
        const category = capability?.category || '组合步骤';
        const status = capability?.availability === 'review' ? '需核验' : '已配置';
        const editing = transformEditingStepIndex === index;
        return `<article class="transform-step-card ${editing ? 'is-editing' : ''}">
            <div class="transform-step-index">${index + 1}</div>
            <div class="transform-step-main">
                <div class="transform-step-title"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(category)}</span><em>${status}</em></div>
                <div class="transform-step-params">${transformParamSummary(step.params || {}) || '<span><small>参数</small>使用默认配置</span>'}</div>
            </div>
            <div class="transform-step-actions">
                <button type="button" class="is-edit" title="${editing ? '收起设置' : '编辑参数'}" onclick="toggleTransformStepEditor(${index})">${editing ? '收起' : '设置'}</button>
                <button type="button" title="上移" onclick="moveTransformStep(${index}, -1)" ${index === 0 ? 'disabled' : ''}>↑</button>
                <button type="button" title="下移" onclick="moveTransformStep(${index}, 1)" ${index === pipeline.length - 1 ? 'disabled' : ''}>↓</button>
                <button type="button" class="is-danger" title="删除" onclick="removeTransformStep(${index})">×</button>
            </div>
            ${editing ? renderTransformStepEditor(index, step) : ''}
        </article>`;
    }).join('');
}

function updateTransformPipeline(pipeline, message = '') {
    const editor = document.getElementById('transform-pipeline');
    editor.value = pipeline.length ? JSON.stringify(pipeline, null, 2) : '';
    syncTransformPipelineCards();
    disposeTransformPreviewChart();
    transformPreviewResult = null;
    transformPreviewPresets = [];
    document.getElementById('transform-result').innerHTML = '';
    if (message) document.getElementById('transform-status').textContent = message;
}

function moveTransformStep(index, delta) {
    let pipeline;
    try { pipeline = readTransformPipeline(); } catch (error) { showToast(error.message, 'error'); return; }
    const target = index + delta;
    if (target < 0 || target >= pipeline.length) return;
    [pipeline[index], pipeline[target]] = [pipeline[target], pipeline[index]];
    if (transformEditingStepIndex === index) transformEditingStepIndex = target;
    else if (transformEditingStepIndex === target) transformEditingStepIndex = index;
    updateTransformPipeline(pipeline, `已调整步骤顺序：第 ${index + 1} 步移动到第 ${target + 1} 步。`);
}

function removeTransformStep(index) {
    let pipeline;
    try { pipeline = readTransformPipeline(); } catch (error) { showToast(error.message, 'error'); return; }
    pipeline.splice(index, 1);
    if (transformEditingStepIndex === index) transformEditingStepIndex = null;
    else if (transformEditingStepIndex > index) transformEditingStepIndex -= 1;
    updateTransformPipeline(pipeline, pipeline.length ? `已删除步骤 ${index + 1}，请重新预览。` : '处理方案已清空。');
}

async function suggestTransformPipeline() {
    const goalInput = document.getElementById('transform-goal');
    const status = document.getElementById('transform-status');
    const recommendationsBox = document.getElementById('transform-recommendations');
    const problemInput = document.getElementById('problem-description');
    const problem = goalInput.value.trim() || problemInput?.value.trim() || '';
    status.textContent = '正在从字段画像与处理目标生成候选流水线…';
    recommendationsBox.innerHTML = '';
    try {
        const response = await fetch('/api/data/transform/suggest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ problem })
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || '生成失败');
        transformRecommendations = data.recommendations || [];
        const profile = data.profile || {};
        status.textContent = `画像：${Number(profile.rows || 0).toLocaleString()} 行，${profile.columns || 0} 列；数值 ${profile.numeric?.length || 0}，类别 ${profile.categorical?.length || 0}，时间 ${profile.datetime?.length || 0}。`;
        if (transformRecommendations.length === 0) {
            recommendationsBox.innerHTML = '<p class="hint">没有足够证据自动生成可靠方案。可从下方操作注册表组合流水线，系统不会猜测字段含义。</p>';
            return;
        }
        recommendationsBox.innerHTML = transformRecommendations.map((item, index) => `
            <article class="transform-recommendation-card">
                <div><strong>${escapeHtml(item.name)}</strong><p>${escapeHtml(item.reason)}</p><small>核验点：${escapeHtml(item.risk)}</small></div>
                <button class="btn btn-sm" onclick="useTransformRecommendation(${index})">载入候选</button>
            </article>`).join('');
        if (!document.getElementById('transform-pipeline').value.trim()) useTransformRecommendation(0);
    } catch (error) {
        status.textContent = '';
        showToast('候选方案生成失败: ' + error.message, 'error');
    }
}

async function runMathematicalDataCompilation() {
    const goal = document.getElementById('transform-goal').value.trim()
        || document.getElementById('problem-description')?.value.trim()
        || '';
    let target = document.getElementById('transform-target').value.trim();
    if (!target) {
        const researchTarget = document.getElementById('research-target')?.value.trim();
        if (researchTarget) target = researchTarget.split(/[,，;；]/)[0].trim().split('.').pop();
    }
    const status = document.getElementById('transform-status');
    status.textContent = '正在编译估计对象、数据粒度与多视图反证…';
    try {
        const semanticText = document.getElementById('transform-semantic-hints')?.value.trim() || '';
        let semanticHints = null;
        if (semanticText) {
            try {
                semanticHints = JSON.parse(semanticText);
            } catch (parseError) {
                throw new Error('字段语义契约不是有效 JSON：' + parseError.message);
            }
            if (!semanticHints || Array.isArray(semanticHints) || typeof semanticHints !== 'object') {
                throw new Error('字段语义契约必须是 JSON 对象');
            }
        }
        const response = await fetch('/api/data/math-compile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                problem: goal,
                target: target || null,
                max_views: 8,
                semantic_hints: semanticHints
            })
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || '数学数据编译失败');
        mathematicalDataCompilation = data.result;
        renderMathematicalDataCompilation(data.result);
        const summary = data.result.summary || {};
        const scopeText = summary.sampled_execution
            ? `覆盖样本 ${Number(summary.audited_rows || 0).toLocaleString()}/${Number(summary.source_rows || 0).toLocaleString()} 行`
            : '完整数据';
        status.textContent = `多视图审计完成（${scopeText}）：${summary.admissible_views || 0}/${summary.candidate_views || 0} 个候选通过，发现 ${summary.direction_reversals || 0} 个方向翻转。`;
    } catch (error) {
        status.textContent = '数学数据编译未完成，当前数据未改变。';
        showToast('数学多视图审计失败: ' + error.message, 'error');
    }
}

function renderMathematicalDataCompilation(result) {
    const box = document.getElementById('math-data-compilation-result');
    const contract = result.contract || {};
    const summary = result.summary || {};
    const views = result.views || [];
    const relationships = (result.conclusion_stress || {}).relationships || [];
    const statusClass = result.status === 'contradicted' ? 'research-fail'
        : (result.status === 'assessed' ? 'research-safe' : 'research-risk');
    let html = `<details class="math-data-section" open><summary>数学数据契约与多视图反证 <span class="${statusClass}">${escapeHtml(result.status || '-')}</span></summary>`;
    html += `<div class="research-metrics"><span><small>目标</small><strong>${escapeHtml(contract.target || '未绑定')}</strong></span><span><small>观测粒度</small><strong>${escapeHtml((contract.observed_grain || []).join(' × ') || '未验证')}</strong><small>${escapeHtml(contract.grain_status || '-')}</small></span><span><small>语义来源</small><strong>${escapeHtml(contract.semantic_contract_source || 'heuristic')}</strong><small>显式 ${Number((contract.semantic_hints_applied || []).length)} 项</small></span><span><small>粒度唯一性</small><strong>${contract.grain_uniqueness == null ? '-' : (contract.grain_uniqueness * 100).toFixed(1) + '%'}</strong></span><span><small>审计行数</small><strong>${Number(summary.audited_rows || contract.profiled_rows || 0).toLocaleString()} / ${Number(summary.source_rows || contract.rows || 0).toLocaleString()}</strong></span><span><small>编译耗时</small><strong>${Number((summary.timing_ms || {}).total || 0).toFixed(1)} ms</strong></span><span><small>候选视图</small><strong>${summary.admissible_views || 0} / ${summary.candidate_views || 0}</strong></span><span><small>方向翻转</small><strong class="${summary.direction_reversals ? 'research-fail' : 'research-safe'}">${summary.direction_reversals || 0}</strong></span></div>`;
    html += `<p class="hint"><strong>估计对象：</strong>${escapeHtml(contract.estimand || '-')}</p>`;
    if ((contract.unresolved || []).length) {
        html += `<p class="research-warning"><strong>未决语义：</strong>${escapeHtml(contract.unresolved.join('；'))}</p>`;
    }
    if (views.length) {
        html += '<h4>候选建模视图</h4><div class="table-wrapper"><table class="data-table math-view-table"><thead><tr><th>视图</th><th>估计对象</th><th>粒度关系</th><th>输出规模</th><th>守恒/泄漏</th><th>处置</th></tr></thead><tbody>';
        views.forEach((view, index) => {
            const viewClass = view.admissible ? 'research-safe' : 'research-fail';
            const conservationFailed = (view.conservation_audit || []).filter(item => item.status === 'fail').length;
            const auditText = `守恒失败 ${conservationFailed}；泄漏 ${escapeHtml(view.leakage_audit?.status || '-')}`;
            const action = (view.pipeline || []).length
                ? `<button class="btn btn-sm" onclick="loadMathematicalDataView(${index})" ${view.admissible ? '' : 'disabled'}>载入流水线</button>`
                : '<span class="hint">基线</span>';
            const admissionText = view.admissible
                ? (summary.sampled_execution ? '样本审计通过' : '完整审计通过')
                : '阻断';
            html += `<tr><td><strong>${escapeHtml(view.name)}</strong><br><small>${escapeHtml(view.purpose)}</small></td><td>${escapeHtml(view.estimand || '-')}</td><td>${escapeHtml(view.row_relation || '-')}<br><small>${escapeHtml((view.output_grain || []).join(' × ') || '-')}</small></td><td>${view.output_shape ? view.output_shape.join('×') : '-'}</td><td>${auditText}</td><td><span class="${viewClass}">${admissionText}</span><br>${action}${(view.blocking_reasons || []).length ? `<small>${escapeHtml(view.blocking_reasons.join('；'))}</small>` : ''}</td></tr>`;
        });
        html += '</tbody></table></div>';
    }
    const stressed = relationships.filter(item => item.status !== 'stable_empirical');
    if (stressed.length) {
        html += '<h4>结论可信度与视图敏感性</h4><div class="table-wrapper"><table class="data-table"><thead><tr><th>关系</th><th>状态</th><th>全局ρ / 95%区间</th><th>FDR q</th><th>效应跨度</th><th>审计解释</th></tr></thead><tbody>';
        stressed.slice(0, 20).forEach(item => {
            const global = (item.contexts || []).find(context => context.view === 'global_complete_case');
            const interval = (global?.confidence_interval_95 || []).join(', ');
            html += `<tr><td>${escapeHtml(item.predictor)} → ${escapeHtml(item.target)}</td><td><span class="${item.status === 'contradicted' ? 'research-fail' : 'research-risk'}">${escapeHtml(item.status)}</span></td><td>${global?.rho ?? '-'}<br><small>[${escapeHtml(interval || '-')} ]</small></td><td>${item.global_fdr_q ?? '-'}</td><td>${item.effect_spread ?? '-'}</td><td>${escapeHtml(item.interpretation || '-')}</td></tr>`;
        });
        html += '</tbody></table></div>';
    }
    (result.findings || []).forEach(item => {
        const findingClass = item.level === 'contradicted' || item.level === 'blocked' ? 'research-warning' : 'hint';
        html += `<p class="${findingClass}"><strong>${escapeHtml(item.message)}</strong><br>${escapeHtml(item.action || '')}</p>`;
    });
    html += '</details>';
    box.innerHTML = html;
}

function loadMathematicalDataView(index) {
    const view = mathematicalDataCompilation?.views?.[index];
    if (!view || !view.admissible || !(view.pipeline || []).length) return;
    updateTransformPipeline(view.pipeline);
    document.getElementById('transform-status').textContent = `已载入“${view.name}”。该视图的估计对象是：${view.estimand}。请预览后再应用。`;
}

function useTransformRecommendation(index) {
    const recommendation = transformRecommendations[index];
    if (!recommendation) return;
    updateTransformPipeline(recommendation.pipeline);
    document.getElementById('transform-status').textContent = `已载入“${recommendation.name}”。请先预览，并核验：${recommendation.risk}`;
}

function isTransformCategoricalBar(result) {
    const encodings = result?.encodings || {};
    const xKind = result?.field_types?.[encodings.x];
    const yKind = result?.field_types?.[encodings.y];
    return result?.chart_type === 'bar'
        && !['numeric', 'datetime'].includes(xKind) && yKind === 'numeric';
}

function resetTransformPreviewView(result) {
    const categoricalBar = isTransformCategoricalBar(result);
    transformPreviewView = {
        orientation: categoricalBar ? 'horizontal' : 'vertical',
        sort: categoricalBar ? 'desc' : 'none',
        topN: categoricalBar ? 15 : 0,
        labels: categoricalBar,
    };
}

function selectTransformPreviewPreset(index) {
    const preset = transformPreviewPresets[index];
    if (!preset?.result) return;
    transformPreviewResult = preset.result;
    resetTransformPreviewView(transformPreviewResult);
    document.querySelectorAll('.transform-preview-preset').forEach((button, buttonIndex) => {
        button.classList.toggle('active', buttonIndex === index);
    });
    const title = document.getElementById('transform-preview-title');
    const reason = document.getElementById('transform-preview-reason');
    const categoricalBar = isTransformCategoricalBar(transformPreviewResult);
    if (title) title.textContent = categoricalBar ? '排名比较' : interactiveVizChartLabel(transformPreviewResult.chart_type);
    if (reason) reason.textContent = transformPreviewResult.reason || '';
    document.getElementById('transform-bar-controls')?.classList.toggle('hidden', !categoricalBar);
    renderTransformPreviewChart(transformPreviewResult);
    syncTransformPreviewToolbar();
}

function renderTransformResult(data, committed) {
    const box = document.getElementById('transform-result');
    const audit = data.audit || [];
    const warnings = data.warnings || [];
    const preview = data.data || {};
    const inputRows = Number(data.input_shape?.[0] || 0);
    const inputColumns = Number(data.input_shape?.[1] || 0);
    const outputRows = Number(data.shape?.[0] || 0);
    const outputColumns = Number(data.shape?.[1] || 0);
    const rowDelta = outputRows - inputRows;
    const columnDelta = outputColumns - inputColumns;
    disposeTransformPreviewChart();
    transformPreviewPresets = data.visual_preview?.presets || [];
    transformPreviewResult = transformPreviewPresets[0]?.result || data.visual_preview || null;
    const previewEncodings = transformPreviewResult?.encodings || {};
    const previewXKind = transformPreviewResult?.field_types?.[previewEncodings.x];
    const categoricalBar = transformPreviewResult?.chart_type === 'bar'
        && !['numeric', 'datetime'].includes(previewXKind);
    resetTransformPreviewView(transformPreviewResult);
    let html = `<div class="transform-result-head"><strong>${committed ? '已提交' : '预览'}：${Number(data.shape?.[0] || 0).toLocaleString()} 行 × ${Number(data.shape?.[1] || 0).toLocaleString()} 列</strong><span>内存 ${(Number(data.memory_bytes || 0) / 1048576).toFixed(2)} MB</span></div>`;
    html += `
        <section class="transform-visual-preview">
            <div class="transform-visual-header">
                <div>
                    <span class="transform-visual-kicker">结果洞察</span>
                    <h4 id="transform-preview-title">${categoricalBar ? '排名比较' : (escapeHtml(interactiveVizChartLabel(transformPreviewResult?.chart_type || '')) || '结果概览')}</h4>
                    <p id="transform-preview-reason">${escapeHtml(transformPreviewResult?.reason || '正在检查结果是否具备可解释的绘图字段。')}</p>
                </div>
                <span class="transform-visual-scope">预览只读 · 最多 1,200 图元</span>
            </div>
            <div class="transform-shape-flow" aria-label="变换前后结构对比">
                <div><small>输入</small><strong>${inputRows.toLocaleString()} 行 · ${inputColumns.toLocaleString()} 列</strong></div>
                <b aria-hidden="true">→</b>
                <div><small>输出</small><strong>${outputRows.toLocaleString()} 行 · ${outputColumns.toLocaleString()} 列</strong></div>
                <div class="transform-shape-delta"><small>变化</small><strong>${rowDelta >= 0 ? '+' : ''}${rowDelta.toLocaleString()} 行 · ${columnDelta >= 0 ? '+' : ''}${columnDelta.toLocaleString()} 列</strong></div>
            </div>
            ${transformPreviewPresets.length > 1 ? `<div class="transform-preview-presets" aria-label="结果图形方案">${transformPreviewPresets.map((preset, index) => `<button type="button" class="transform-preview-preset ${index === 0 ? 'active' : ''}" onclick="selectTransformPreviewPreset(${index})">${escapeHtml(preset.label)}</button>`).join('')}</div>` : ''}
            <div id="transform-bar-controls" class="transform-chart-toolbar ${categoricalBar ? '' : 'hidden'}" aria-label="图形显示选项">
                <div class="transform-segmented">
                    <button type="button" data-orientation="horizontal" onclick="setTransformPreviewView('orientation','horizontal')">横向排名</button>
                    <button type="button" data-orientation="vertical" onclick="setTransformPreviewView('orientation','vertical')">纵向比较</button>
                </div>
                <label>排序<select id="transform-preview-sort" onchange="setTransformPreviewView('sort',this.value)"><option value="desc">从高到低</option><option value="asc">从低到高</option><option value="none">原始顺序</option></select></label>
                <label>显示<select id="transform-preview-topn" onchange="setTransformPreviewView('topN',this.value)"><option value="10">前 10</option><option value="15" selected>前 15</option><option value="30">前 30</option><option value="0">全部</option></select></label>
                <label class="transform-label-toggle"><input id="transform-preview-labels" type="checkbox" checked onchange="setTransformPreviewView('labels',this.checked)"> 显示数值</label>
            </div>
            <div class="transform-chart-stage">
                <div id="transform-preview-chart" class="transform-preview-chart" role="img" aria-label="变换结果自动图"></div>
                <aside id="transform-preview-insight" class="transform-preview-insight" aria-live="polite"></aside>
            </div>
            <div id="transform-preview-chart-note" class="transform-preview-chart-note"></div>
        </section>`;
    if (warnings.length) {
        html += '<ul class="transform-warning-list">' + warnings.map(item => `<li>${escapeHtml(item)}</li>`).join('') + '</ul>';
    }
    if (audit.length) {
        html += `<details class="transform-result-details"><summary>查看 ${audit.length} 步执行审计</summary><div class="table-wrapper"><table class="data-table transform-audit-table"><thead><tr><th>步骤</th><th>操作</th><th>输入</th><th>输出</th><th>新增字段</th><th>删除字段</th><th>耗时</th></tr></thead><tbody>`;
        audit.forEach(item => {
            html += `<tr><td>${item.step}</td><td>${escapeHtml(item.operation)}</td><td>${item.input_shape.join('×')}</td><td>${item.output_shape.join('×')}</td><td>${escapeHtml((item.columns_added || []).join('、') || '-')}</td><td>${escapeHtml((item.columns_removed || []).join('、') || '-')}</td><td>${item.elapsed_ms} ms</td></tr>`;
        });
        html += '</tbody></table></div></details>';
    }
    if (preview.columns && preview.preview) {
        html += `<details class="transform-result-details"><summary>查看结果抽样（前 ${Math.min(10, preview.preview.length)} 行）</summary><div class="table-wrapper"><table class="data-table"><thead><tr>`;
        preview.columns.forEach(column => { html += `<th>${escapeHtml(column)}</th>`; });
        html += '</tr></thead><tbody>';
        preview.preview.slice(0, 10).forEach(row => {
            html += '<tr>';
            preview.columns.forEach(column => {
                const value = row[column];
                html += `<td>${value === null ? '<span class="transform-null">NULL</span>' : escapeHtml(String(value))}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table></div></details>';
    }
    box.innerHTML = html;
    renderTransformPreviewChart(transformPreviewResult);
    syncTransformPreviewToolbar();
}

function disposeTransformPreviewChart() {
    if (!transformPreviewChart) return;
    transformPreviewChart.dispose();
    transformPreviewChart = null;
}

function setTransformPreviewView(key, value) {
    if (key === 'topN') value = Math.max(0, Number(value) || 0);
    if (key === 'labels') value = Boolean(value);
    transformPreviewView[key] = value;
    renderTransformPreviewChart(transformPreviewResult);
    syncTransformPreviewToolbar();
}

function syncTransformPreviewToolbar() {
    document.querySelectorAll('.transform-segmented button').forEach(button => {
        button.classList.toggle('active', button.dataset.orientation === transformPreviewView.orientation);
    });
    const sort = document.getElementById('transform-preview-sort');
    const topN = document.getElementById('transform-preview-topn');
    const labels = document.getElementById('transform-preview-labels');
    if (sort) sort.value = transformPreviewView.sort;
    if (topN) topN.value = String(transformPreviewView.topN);
    if (labels) labels.checked = Boolean(transformPreviewView.labels);
}

function formatTransformChartValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value ?? '-');
    return Math.abs(number) >= 1000
        ? number.toLocaleString(undefined, { maximumFractionDigits: 2 })
        : Number(number.toPrecision(5)).toString();
}

function renderTransformPreviewInsight(records, result) {
    const box = document.getElementById('transform-preview-insight');
    if (!box) return;
    const enc = result?.encodings || {};
    const numericPairs = records.map(record => ({
        label: String(record[enc.x] ?? '缺失'),
        value: Number(record[enc.y]),
    })).filter(item => Number.isFinite(item.value));
    if (!numericPairs.length) {
        box.innerHTML = '<span>一眼结论</span><h5>暂无稳定数值摘要</h5><p>请结合下方抽样表核验字段内容。</p>';
        return;
    }
    if (result.chart_type === 'bar') {
        const ranked = [...numericPairs].sort((left, right) => right.value - left.value);
        const values = ranked.map(item => item.value).sort((left, right) => left - right);
        const top = ranked[0];
        const bottom = ranked[ranked.length - 1];
        const median = values.length % 2
            ? values[Math.floor(values.length / 2)]
            : (values[values.length / 2 - 1] + values[values.length / 2]) / 2;
        box.innerHTML = `<span>一眼结论</span><h5>${escapeHtml(top.label)} 最高</h5>
            <strong>${escapeHtml(formatTransformChartValue(top.value))}</strong>
            <p>中位数 ${escapeHtml(formatTransformChartValue(median))}；最高与最低相差 ${escapeHtml(formatTransformChartValue(top.value - bottom.value))}。</p>
            <small>这是结果描述，不代表差异显著或具有因果关系。</small>`;
        return;
    }
    if (['line', 'area'].includes(result.chart_type)) {
        const first = numericPairs[0];
        const last = numericPairs[numericPairs.length - 1];
        const delta = last.value - first.value;
        box.innerHTML = `<span>一眼结论</span><h5>${delta >= 0 ? '期末高于期初' : '期末低于期初'}</h5>
            <strong>${delta >= 0 ? '+' : ''}${escapeHtml(formatTransformChartValue(delta))}</strong>
            <p>从 ${escapeHtml(formatTransformChartValue(first.value))} 变化到 ${escapeHtml(formatTransformChartValue(last.value))}，共 ${numericPairs.length} 个可见点。</p>
            <small>趋势仍需检查季节性、异常点和时间窗口敏感性。</small>`;
        return;
    }
    box.innerHTML = `<span>一眼结论</span><h5>联合分布</h5><strong>${numericPairs.length.toLocaleString()} 点</strong><p>用于寻找方向、分群、非线性与异常点。</p><small>形态只提示关联，不证明因果。</small>`;
}

function renderTransformPreviewChart(result) {
    const chartDom = document.getElementById('transform-preview-chart');
    const note = document.getElementById('transform-preview-chart-note');
    if (!chartDom || !note) return;
    let records = [...(result?.records || [])];
    if (!result?.available || !records.length || typeof echarts === 'undefined') {
        chartDom.classList.add('is-empty');
        chartDom.innerHTML = `<div><strong>当前没有可解释的自动图</strong><span>${escapeHtml(result?.reason || '缺少可绘制字段或记录。')}</span></div>`;
        note.textContent = '表格抽样与步骤审计仍可在下方核验。';
        return;
    }

    const enc = result.encodings || {};
    const xKind = result.field_types?.[enc.x];
    const yKind = result.field_types?.[enc.y];
    const categoricalBar = result.chart_type === 'bar'
        && !['numeric', 'datetime'].includes(xKind) && yKind === 'numeric';
    if (categoricalBar && transformPreviewView.sort !== 'none') {
        const direction = transformPreviewView.sort === 'asc' ? 1 : -1;
        records.sort((left, right) => direction * (Number(left[enc.y]) - Number(right[enc.y])));
    }
    if (categoricalBar && transformPreviewView.topN > 0) {
        records = records.slice(0, transformPreviewView.topN);
    }
    const horizontal = categoricalBar && transformPreviewView.orientation === 'horizontal';
    const numericColor = enc.color && result.field_types?.[enc.color] === 'numeric';
    const groups = groupInteractiveVizRecords(records, enc.color, numericColor, 10);
    const chartType = result.chart_type === 'area' ? 'line' : result.chart_type;
    const tooltipFields = [enc.x, enc.y, enc.color, enc.size, ...(enc.tooltip || [])];
    const series = Array.from(groups.entries()).map(([name, groupRecords]) => ({
        name,
        type: chartType,
        data: groupRecords.map(record => ({
            value: horizontal
                ? [record[enc.y], record[enc.x]]
                : [record[enc.x], record[enc.y]],
            raw: record,
        })),
        showSymbol: ['line', 'area'].includes(result.chart_type) ? groupRecords.length < 240 : true,
        symbolSize: result.chart_type === 'scatter' ? 9 : 7,
        areaStyle: result.chart_type === 'area' ? { opacity: 0.16 } : undefined,
        lineStyle: { width: 2.4 },
        itemStyle: result.chart_type === 'bar'
            ? { borderRadius: horizontal ? [2, 7, 7, 2] : [7, 7, 2, 2], opacity: 0.9 }
            : { borderColor: '#fff', borderWidth: result.chart_type === 'scatter' ? 1 : 0, opacity: 0.88 },
        label: result.chart_type === 'bar' ? {
            show: Boolean(transformPreviewView.labels),
            position: horizontal ? 'right' : 'top',
            color: '#345364',
            fontWeight: 700,
            fontSize: 10,
            formatter: params => formatTransformChartValue(params.value[horizontal ? 0 : 1]),
        } : undefined,
        barMaxWidth: 44,
        large: result.chart_type === 'scatter' && groupRecords.length > 800,
        progressive: 800,
        emphasis: { focus: 'series' },
    }));
    const categoryLevels = new Set(records.map(record => String(record[enc.x]))).size;
    const showSlider = categoricalBar && categoryLevels > 16;
    chartDom.style.height = horizontal
        ? `${Math.min(540, Math.max(280, categoryLevels * 31 + 75))}px`
        : '350px';
    transformPreviewChart = echarts.getInstanceByDom(chartDom) || echarts.init(chartDom);
    transformPreviewChart.clear();
    transformPreviewChart.setOption({
        animation: records.length < 600,
        color: ['#1d6f8a', '#31a28b', '#e7a23b', '#d95d63', '#725cc5', '#4b91ca', '#93a83d', '#c4679b'],
        textStyle: { fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif" },
        legend: groups.size > 1 ? { type: 'scroll', top: 4, left: 'center', itemWidth: 14, itemHeight: 8, textStyle: { color: '#516679', fontSize: 10 } } : undefined,
        toolbox: { right: 8, top: 0, feature: { saveAsImage: { title: '保存图片' }, dataZoom: { title: { zoom: '缩放', back: '还原缩放' } }, restore: { title: '还原' } }, iconStyle: { borderColor: '#71879a' } },
        tooltip: {
            trigger: 'item', confine: true, backgroundColor: 'rgba(12,30,49,.95)', borderWidth: 0,
            textStyle: { color: '#edf6fb', fontSize: 11 },
            formatter: params => interactiveVizTooltipFormatter(params, tooltipFields),
        },
        grid: { left: horizontal ? 30 : 50, right: horizontal && transformPreviewView.labels ? 64 : 25, top: groups.size > 1 ? 48 : 32, bottom: 42, containLabel: true },
        xAxis: horizontal ? {
            type: 'value', name: interactiveVizDisplayField(enc.y), nameLocation: 'middle', nameGap: 28,
            axisLabel: { color: '#718291', fontSize: 9 }, axisLine: { show: false },
            splitLine: { lineStyle: { color: '#e7edf2', type: 'dashed' } },
        } : {
            type: xKind === 'numeric' ? 'value' : (xKind === 'datetime' ? 'time' : 'category'),
            name: interactiveVizDisplayField(enc.x), nameLocation: 'middle', nameGap: showSlider ? 54 : 32,
            axisLabel: { hideOverlap: true, rotate: !['numeric', 'datetime'].includes(xKind) ? 20 : 0, color: '#63778a', fontSize: 10 },
            axisLine: { lineStyle: { color: '#a8b7c4' } }, splitLine: { show: xKind === 'numeric', lineStyle: { color: '#e9eef2', type: 'dashed' } },
        },
        yAxis: horizontal ? {
            type: 'category', inverse: transformPreviewView.sort === 'desc',
            axisLabel: { width: 130, overflow: 'truncate', color: '#425c6b', fontSize: 10 },
            axisLine: { show: false }, axisTick: { show: false },
        } : {
            type: yKind === 'numeric' ? 'value' : 'category',
            name: interactiveVizDisplayField(enc.y), nameLocation: 'middle', nameGap: 44,
            axisLabel: { hideOverlap: true, color: '#63778a', fontSize: 10 },
            axisLine: { show: false }, splitLine: { lineStyle: { color: '#e7edf2', type: 'dashed' } },
        },
        dataZoom: showSlider ? [
            { type: 'inside', yAxisIndex: 0, filterMode: 'filter' },
            { type: 'slider', yAxisIndex: 0, right: 2, width: 15, filterMode: 'filter' },
        ] : [{ type: 'inside', [horizontal ? 'yAxisIndex' : 'xAxisIndex']: 0, filterMode: 'filter' }],
        series,
    }, true);
    const audit = result.audit || {};
    const aggregation = audit.aggregation || {};
    const warnings = result.warnings || [];
    const chartLabel = horizontal ? '横向排名' : interactiveVizChartLabel(result.chart_type);
    const aggregationLabel = ({ none: '逐行展示', count: '计数', sum: '求和', mean: '均值', median: '中位数', min: '最小值', max: '最大值' })[aggregation.function] || aggregation.function || '逐行展示';
    note.innerHTML = `<span>${escapeHtml(chartLabel)}</span><span>图元 ${Number(audit.output_rows || records.length).toLocaleString()} / 结果 ${Number(audit.source_rows || 0).toLocaleString()} 行</span><span>口径 ${escapeHtml(aggregationLabel)}</span>${warnings.length ? `<em>⚠ ${escapeHtml(warnings[0])}</em>` : ''}`;
    renderTransformPreviewInsight(records, result);
}

async function executeTableTransformation(commit) {
    const status = document.getElementById('transform-status');
    let pipeline;
    try {
        pipeline = readTransformPipeline();
    } catch (error) {
        showToast(error.message, 'error');
        return;
    }
    const actionButton = document.getElementById(commit ? 'transform-apply-btn' : 'transform-preview-btn');
    if (actionButton) actionButton.disabled = true;
    status.textContent = commit ? '正在用有界样本预检字段绑定与步骤组合…' : '正在完整数据上执行预览（不会修改当前表）…';
    try {
        if (commit) {
            const validationResponse = await fetch('/api/data/transform/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pipeline })
            });
            const validation = await validationResponse.json();
            if (!validation.success) {
                renderTransformValidationFailure(validation);
                status.textContent = '应用已阻止：流水线字段或步骤组合与当前表不兼容，数据保持不变。';
                showToast('应用前校验未通过，请按页面建议修正流水线', 'error');
                return;
            }
            status.textContent = `预检通过（${Number(validation.sampled_rows || 0).toLocaleString()} 行）：正在事务式执行完整流水线…`;
        }
        const endpoint = commit ? '/api/data/transform/apply' : '/api/data/transform/preview';
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pipeline })
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || '流水线执行失败');
        renderTransformResult(data, commit);
        status.textContent = commit
            ? `已提交 ${data.audit?.length || 0} 步变换；后续分析与建模将使用该结果。${data.undo_available ? '可撤销。' : '未保存撤销快照。'}`
            : `预览完成，共 ${data.audit?.length || 0} 步；确认字段和行数后再应用。`;
        if (commit) {
            uploadedData = data.data;
            showUploadResult(data);
            document.getElementById('data-status').textContent = `变换结果: ${data.shape[0]}行×${data.shape[1]}列 · ${data.audit.length}步可审计流水线`;
            document.getElementById('data-status').classList.add('loaded');
            populateTargetOptions(data.data.columns, null);
            showToast('数据流水线已应用到后续建模');
        }
    } catch (error) {
        status.textContent = '流水线未提交，当前数据保持不变。';
        showToast('数据变换失败: ' + error.message, 'error');
    } finally {
        if (actionButton) actionButton.disabled = false;
    }
}

function renderTransformValidationFailure(validation) {
    const box = document.getElementById('transform-result');
    disposeTransformPreviewChart();
    const columns = validation.current_columns || uploadedData?.columns || [];
    transformValidationSuggestion = Array.isArray(validation.suggested_pipeline) ? validation.suggested_pipeline : null;
    const repairButton = transformValidationSuggestion !== null
        ? `<button class="btn btn-sm transform-repair-btn" onclick="applyTransformValidationSuggestion()">${transformValidationSuggestion.length ? `移除第 ${Number(validation.invalid_step || 0)} 步` : '清空不适用步骤'}</button>`
        : '';
    box.innerHTML = `
        <section class="transform-preflight-failure" role="alert">
            <div class="transform-preflight-icon">!</div>
            <div>
                <span class="transform-preflight-kicker">应用前校验未通过</span>
                <h4>${escapeHtml(validation.error || '流水线与当前表不兼容')}</h4>
                <p>${escapeHtml(validation.action || '请重新绑定字段后再执行。')}</p>
                <div class="transform-current-columns"><strong>当前可用字段</strong>${columns.map(column => `<span>${escapeHtml(String(column))}</span>`).join('')}</div>
                ${repairButton}
            </div>
        </section>`;
    box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function applyTransformValidationSuggestion() {
    if (!Array.isArray(transformValidationSuggestion)) return;
    const message = transformValidationSuggestion.length
        ? '已移除不兼容步骤；请重新预览或应用。'
        : '不兼容流水线已清空；请从已绑定当前字段的操作注册表重新选择。';
    updateTransformPipeline(transformValidationSuggestion, message);
    transformValidationSuggestion = null;
}

async function undoTableTransformation() {
    const status = document.getElementById('transform-status');
    try {
        const response = await fetch('/api/data/transform/undo', { method: 'POST' });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || '撤销失败');
        uploadedData = data.data;
        showUploadResult(data);
        document.getElementById('data-status').textContent = `已撤销数据变换: ${data.shape[0]}行×${data.shape[1]}列`;
        populateTargetOptions(data.data.columns, null);
        disposeTransformPreviewChart();
        document.getElementById('transform-result').innerHTML = '';
        status.textContent = `已恢复上一版本。${data.undo_available ? '仍可继续撤销。' : ''}`;
        showToast('已撤销上次数据变换');
    } catch (error) {
        showToast(error.message, 'error');
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
    const leftKeySelect = document.getElementById('join-left-on');
    const rightKeySelect = document.getElementById('join-right-on');
    
    if (!leftVal || !rightVal) {
        leftKeySelect.innerHTML = '<option value="">先选择左右表</option>';
        rightKeySelect.innerHTML = '<option value="">先选择左右表</option>';
        return;
    }
    
    try {
        const left = JSON.parse(leftVal);
        const right = JSON.parse(rightVal);
        const [leftResponse, rightResponse] = await Promise.all([
            fetch('/api/upload/schema', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: left })
            }),
            fetch('/api/upload/schema', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: right })
            })
        ]);
        const leftSchema = await leftResponse.json();
        const rightSchema = await rightResponse.json();
        if (!leftSchema.success || !rightSchema.success) {
            throw new Error(leftSchema.error || rightSchema.error || '字段读取失败');
        }
        const leftColumns = leftSchema.columns || [];
        const rightColumns = rightSchema.columns || [];
        const common = leftColumns.filter(column => rightColumns.includes(column));
        const defaultKey = common[0] || null;
        leftKeySelect.innerHTML = leftColumns.map(column => `<option value="${escapeHtml(column)}" ${column === defaultKey ? 'selected' : ''}>${escapeHtml(column)}</option>`).join('');
        rightKeySelect.innerHTML = rightColumns.map(column => `<option value="${escapeHtml(column)}" ${column === defaultKey ? 'selected' : ''}>${escapeHtml(column)}</option>`).join('');
        if (!defaultKey) {
            showToast('两表没有同名字段，请分别选择语义对应的左右键');
        }
    } catch (e) {
        leftKeySelect.innerHTML = '<option value="">字段读取失败</option>';
        rightKeySelect.innerHTML = '<option value="">字段读取失败</option>';
        showToast('关联字段读取失败: ' + e.message, 'error');
    }
}

async function executeJoin() {
    const leftVal = document.getElementById('join-left').value;
    const rightVal = document.getElementById('join-right').value;
    const leftOn = Array.from(document.getElementById('join-left-on').selectedOptions).map(option => option.value).filter(Boolean);
    const rightOn = Array.from(document.getElementById('join-right-on').selectedOptions).map(option => option.value).filter(Boolean);
    const how = document.getElementById('join-how').value;
    const keyType = document.getElementById('join-key-type').value;
    const validate = document.getElementById('join-validate').value;
    
    if (!leftVal || !rightVal || leftOn.length === 0 || leftOn.length !== rightOn.length) {
        showToast('请选择左右表，并确保左右关联键数量一致', 'error');
        return;
    }
    
    try {
        const res = await fetch('/api/upload/join', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                left: JSON.parse(leftVal),
                right: JSON.parse(rightVal),
                left_on: leftOn,
                right_on: rightOn,
                how: how,
                key_type: keyType,
                validate: validate
            })
        });
        const data = await res.json();
        if (data.success) {
            uploadedData = data.data;
            showUploadResult(data);
            const diagnostics = data.join_diagnostics || {};
            const relationLabels = { one_to_one: '一对一', one_to_many: '一对多', many_to_one: '多对一', many_to_many: '多对多' };
            document.getElementById('data-status').textContent = `关联结果: ${data.shape[0]}行×${data.shape[1]}列 · ${relationLabels[diagnostics.inferred_relation] || '键关系未知'} · 膨胀${Number(diagnostics.expansion_ratio || 0).toFixed(2)}倍`;
            populateTargetOptions(data.data.columns, data.target_hint);
            showToast(`关联成功！${data.shape[0]}行×${data.shape[1]}列；${relationLabels[diagnostics.inferred_relation] || ''}`);
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
    df.columns.forEach(c => html += `<th>${escapeHtml(String(c))}</th>`);
    html += '</tr></thead><tbody>';
    df.preview.forEach(row => {
        html += '<tr>';
        df.columns.forEach(c => html += `<td>${row[c] !== null ? escapeHtml(String(row[c])) : '<span style="color:#999">NULL</span>'}</td>`);
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    preview.innerHTML = html;
    document.getElementById('upload-result').classList.remove('hidden');
    loadTransformCapabilities();
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
        // 独立于模型训练的多维交互图形工作台
        loadInteractiveVisualizationSchema();

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

// ==================== 多维交互图形工作台 ====================

function interactiveVizField(name) {
    return (interactiveVizSchema?.fields || []).find(field => field.name === name) || null;
}

function interactiveVizDisplayField(name) {
    if (name === '__count__') return '记录数';
    const value = name || '';
    const suffixes = {
        '_sum': '（求和）', '_mean': '（均值）', '_median': '（中位数）',
        '_min': '（最小值）', '_max': '（最大值）', '_count': '（计数）',
    };
    for (const [suffix, label] of Object.entries(suffixes)) {
        if (value.endsWith(suffix)) return value.slice(0, -suffix.length) + label;
    }
    return value;
}

function interactiveVizChartLabel(chartType) {
    return ({ scatter: '散点图', line: '折线图', area: '面积图', bar: '柱状图', parallel: '平行坐标' })[chartType]
        || chartType;
}

function interactiveVizFieldOptions(fields, includeEmpty = true) {
    const empty = includeEmpty ? '<option value="">不使用</option>' : '';
    const labels = {
        measure: '度量', time: '时间', dimension: '分组维度',
        label: '文本标签', identifier: '编码（不建议作为轴）',
    };
    return empty + fields.map(field =>
        `<option value="${escapeHtml(field.name)}">${escapeHtml(field.name)} · ${labels[field.semantic_role] || field.kind}</option>`
    ).join('');
}

function orderInteractiveVizFields(fields) {
    const priority = { time: 0, measure: 1, dimension: 2, label: 3, identifier: 4 };
    return [...fields].sort((left, right) =>
        (priority[left.semantic_role] ?? 9) - (priority[right.semantic_role] ?? 9)
        || String(left.name).localeCompare(String(right.name), 'zh-CN')
    );
}

function setInteractiveVizValue(id, value) {
    const element = document.getElementById(id);
    if (!element) return;
    const wanted = value == null ? '' : String(value);
    if (Array.from(element.options || []).some(option => option.value === wanted)) {
        element.value = wanted;
    }
}

function populateInteractiveVizDatasetSources() {
    const select = document.getElementById('viz-dataset-source');
    if (!select) return;
    const options = [];
    let activeValue = '';
    uploadedFiles.forEach((file, fileIndex) => {
        const sheets = file.sheets && file.sheets.length ? file.sheets : [null];
        sheets.forEach(sheetName => {
            const value = JSON.stringify([fileIndex, sheetName]);
            const label = `${file.filename}${sheetName ? ` · ${sheetName}` : ''}`;
            options.push(`<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`);
            if (file.is_active && (file.active_sheet || null) === sheetName) activeValue = value;
        });
    });
    select.innerHTML = options.length ? options.join('') : '<option value="">当前数据表</option>';
    if (activeValue) select.value = activeValue;
    select.disabled = false;
}

async function switchInteractiveVisualizationDataset() {
    const select = document.getElementById('viz-dataset-source');
    if (!select?.value) return;
    let source;
    try {
        source = JSON.parse(select.value);
    } catch (_) {
        setInteractiveVizConfigNote(['数据表选择状态无效，请重新选择'], true);
        return;
    }
    select.disabled = true;
    const status = document.getElementById('interactive-viz-status');
    status.textContent = '正在切换数据表并重新分析字段语义…';
    const switched = await selectSheet(Number(source[0]), source[1]);
    if (!switched) select.disabled = false;
}

function setInteractiveVizConfigNote(messages = [], isError = false) {
    const note = document.getElementById('interactive-viz-config-note');
    if (!note) return;
    note.classList.toggle('hidden', messages.length === 0);
    note.classList.toggle('is-error', isError);
    note.textContent = messages.join('；');
}

function configureInteractiveVizCapabilities(fields) {
    const enabledCharts = new Set(interactiveVizSchema?.capability?.enabled_charts || ['bar']);
    const chartSelect = document.getElementById('viz-chart-type');
    Array.from(chartSelect.options).forEach(option => {
        if (option.value === 'auto') return;
        option.disabled = !enabledCharts.has(option.value);
        option.title = option.disabled ? '当前数据缺少该图形所需的数学变量角色' : '';
    });
    const roles = role => fields.filter(field => field.semantic_role === role);
    const intentAvailability = {
        auto: true,
        composition: roles('dimension').length > 0,
        comparison: roles('dimension').length > 0 && roles('measure').length > 0,
        relationship: roles('measure').length >= 2,
        trend: roles('time').length > 0 && roles('measure').length > 0,
        custom: true,
    };
    Array.from(document.getElementById('viz-intent').options).forEach(option => {
        option.disabled = !intentAvailability[option.value];
    });
    syncInteractiveVizIntentButtons();
}

function syncInteractiveVizIntentButtons() {
    const select = document.getElementById('viz-intent');
    if (!select) return;
    document.querySelectorAll('[data-viz-intent]').forEach(button => {
        const option = Array.from(select.options).find(item => item.value === button.dataset.vizIntent);
        button.disabled = Boolean(option?.disabled);
        button.classList.toggle('is-active', button.dataset.vizIntent === select.value);
        button.setAttribute('aria-pressed', button.dataset.vizIntent === select.value ? 'true' : 'false');
        button.title = button.disabled ? '当前数据缺少完成这个分析问题所需的变量角色' : '';
    });
}

function selectInteractiveVizIntent(intent) {
    const select = document.getElementById('viz-intent');
    const option = Array.from(select?.options || []).find(item => item.value === intent);
    if (!select || option?.disabled) return;
    select.value = intent;
    syncInteractiveVizIntentButtons();
    applyInteractiveVizIntent();
}

function validateInteractiveVizControls() {
    const chart = document.getElementById('viz-chart-type')?.value;
    const aggregation = document.getElementById('viz-aggregation')?.value;
    const x = interactiveVizField(document.getElementById('viz-x')?.value);
    const y = interactiveVizField(document.getElementById('viz-y')?.value);
    const color = document.getElementById('viz-color')?.value;
    const facet = document.getElementById('viz-facet')?.value;
    const errors = [];
    if (!x && chart !== 'parallel') errors.push('请选择X轴');
    if (x && y && x.name === y.name) errors.push('X轴和Y轴不能使用同一字段');
    if (facet && [x?.name, y?.name, color].includes(facet)) {
        errors.push('分面不能重复使用已经绑定到轴或颜色的字段');
    }
    if (aggregation === 'count') {
        if (chart !== 'bar' && chart !== 'auto') errors.push('计数聚合只能使用柱状图');
        if (y) errors.push('计数模式自动使用“记录数”，不能另选Y轴');
        if (x && ['label', 'identifier'].includes(x.semantic_role)
            && Number(x.unique_count_profiled || 0) > 20
            && Number(x.unique_rate_profiled || 0) > 0.5) {
            errors.push(`“${x.name}”几乎一行一值，计数后只会得到一排1；请选择分组维度`);
        }
    } else {
        if (['scatter', 'line', 'area', 'bar'].includes(chart) && !y) {
            errors.push(`${interactiveVizChartLabel(chart)}需要选择数值Y轴`);
        }
        if (y && y.semantic_role !== 'measure') errors.push('Y轴必须是真实数值度量');
        if (['line', 'area'].includes(chart) && x && !['time', 'measure'].includes(x.semantic_role)) {
            errors.push(`${interactiveVizChartLabel(chart)}的X轴必须是时间或有序数值`);
        }
    }
    return errors;
}

function synchronizeInteractiveVizControls() {
    const notes = [];
    const aggregation = document.getElementById('viz-aggregation');
    const chart = document.getElementById('viz-chart-type');
    const x = document.getElementById('viz-x');
    const y = document.getElementById('viz-y');
    const color = document.getElementById('viz-color');
    const facet = document.getElementById('viz-facet');
    const countMode = aggregation.value === 'count';
    y.disabled = countMode;
    document.getElementById('viz-y-label').textContent = countMode ? 'Y · 自动：记录数' : 'Y · 响应/度量';
    if (countMode) {
        if (y.value) {
            y.value = '';
            notes.push('计数模式已清除Y轴，统一使用记录数');
        }
        if (!['bar', 'auto'].includes(chart.value)) {
            chart.value = 'bar';
            notes.push('计数模式已切换为柱状图');
        }
    }
    Array.from(y.options).forEach(option => {
        option.disabled = Boolean(option.value && option.value === x.value);
    });
    if (y.value && y.value === x.value) {
        y.value = '';
        notes.push('已清除与X轴重复的Y轴');
    }
    const facetConflicts = new Set([x.value, y.value, color.value].filter(Boolean));
    Array.from(facet.options).forEach(option => {
        option.disabled = Boolean(option.value && facetConflicts.has(option.value));
    });
    if (facet.value && facetConflicts.has(facet.value)) {
        facet.value = '';
        notes.push('已清除与轴或颜色重复的分面');
    }
    const selectedChartOption = chart.selectedOptions[0];
    if (selectedChartOption?.disabled) {
        chart.value = 'bar';
        notes.push('当前数据不满足所选图形的数学条件，已切换为柱状图');
    }
    const errors = validateInteractiveVizControls();
    setInteractiveVizConfigNote(errors.length ? errors : notes, errors.length > 0);
    syncInteractiveVizIntentButtons();
    return { notes, errors };
}

function onInteractiveVizControlChange() {
    setInteractiveVizValue('viz-intent', 'custom');
    syncInteractiveVizIntentButtons();
    const validation = synchronizeInteractiveVizControls();
    if (!validation.errors.length) scheduleInteractiveVisualization();
}

function applyInteractiveVizIntent() {
    if (!interactiveVizSchema) return;
    const intent = document.getElementById('viz-intent').value;
    syncInteractiveVizIntentButtons();
    if (intent === 'auto') {
        applyInteractiveVizRecommendation(true);
        return;
    }
    if (intent === 'custom') {
        synchronizeInteractiveVizControls();
        return;
    }
    const fields = interactiveVizSchema.fields || [];
    const measures = fields.filter(field => field.semantic_role === 'measure');
    const times = fields.filter(field => field.semantic_role === 'time');
    const dimensions = fields.filter(field => field.semantic_role === 'dimension');
    const compactDimension = dimensions.find(field => Number(field.unique_count_profiled || 0) <= 12);
    ['color', 'size', 'facet', 'animation'].forEach(channel => setInteractiveVizValue(`viz-${channel}`, ''));
    setInteractiveVizValue('viz-time-unit', 'none');
    if (intent === 'composition') {
        setInteractiveVizValue('viz-chart-type', 'bar');
        setInteractiveVizValue('viz-x', dimensions[0]?.name);
        setInteractiveVizValue('viz-y', '');
        setInteractiveVizValue('viz-aggregation', 'count');
    } else if (intent === 'comparison') {
        setInteractiveVizValue('viz-chart-type', 'bar');
        setInteractiveVizValue('viz-x', dimensions[0]?.name);
        setInteractiveVizValue('viz-y', measures[0]?.name);
        setInteractiveVizValue('viz-aggregation', 'mean');
    } else if (intent === 'relationship') {
        setInteractiveVizValue('viz-chart-type', 'scatter');
        setInteractiveVizValue('viz-x', measures[0]?.name);
        setInteractiveVizValue('viz-y', measures[1]?.name);
        setInteractiveVizValue('viz-color', compactDimension?.name || '');
        setInteractiveVizValue('viz-aggregation', 'none');
    } else if (intent === 'trend') {
        setInteractiveVizValue('viz-chart-type', 'line');
        setInteractiveVizValue('viz-x', times[0]?.name);
        setInteractiveVizValue('viz-y', measures[0]?.name);
        setInteractiveVizValue('viz-color', compactDimension?.name || '');
        setInteractiveVizValue('viz-aggregation', 'mean');
        setInteractiveVizValue('viz-time-unit', 'day');
    }
    const validation = synchronizeInteractiveVizControls();
    if (!validation.errors.length) runInteractiveVisualization();
}

async function loadInteractiveVisualizationSchema() {
    const card = document.getElementById('interactive-viz-card');
    if (!card) return;
    const status = document.getElementById('interactive-viz-status');
    const previousSignature = (interactiveVizSchema?.fields || []).map(field => field.name).join('\u0001');
    const previous = interactiveVizSchema ? {
        intent: document.getElementById('viz-intent')?.value,
        chart: document.getElementById('viz-chart-type')?.value,
        x: document.getElementById('viz-x')?.value,
        y: document.getElementById('viz-y')?.value,
        color: document.getElementById('viz-color')?.value,
        size: document.getElementById('viz-size')?.value,
        facet: document.getElementById('viz-facet')?.value,
        animation: document.getElementById('viz-animation')?.value,
        aggregation: document.getElementById('viz-aggregation')?.value,
        timeUnit: document.getElementById('viz-time-unit')?.value,
        details: Array.from(document.getElementById('viz-detail-fields')?.selectedOptions || []).map(option => option.value),
    } : null;
    status.textContent = '正在读取字段类型、范围与推荐编码…';
    try {
        const response = await fetch('/api/visualization/explore/schema');
        const payload = await response.json();
        if (!payload.success) throw new Error(payload.error || '字段画像失败');
        interactiveVizSchema = payload.schema;
        populateInteractiveVizDatasetSources();
        const fields = orderInteractiveVizFields(interactiveVizSchema.fields || []);
        const measures = fields.filter(field => field.semantic_role === 'measure');
        const colorFields = fields.filter(field => field.channel_suitability?.color)
            .concat(fields.filter(field => !field.channel_suitability?.color));
        const facetFields = fields.filter(field => field.channel_suitability?.facet);
        const animationFields = fields.filter(field => field.channel_suitability?.animation);
        const categoryFields = fields.filter(field => ['dimension', 'label', 'identifier'].includes(field.semantic_role));
        document.getElementById('viz-x').innerHTML = interactiveVizFieldOptions(fields, false);
        document.getElementById('viz-y').innerHTML = interactiveVizFieldOptions(measures, true);
        document.getElementById('viz-color').innerHTML = interactiveVizFieldOptions(colorFields, true);
        document.getElementById('viz-size').innerHTML = interactiveVizFieldOptions(measures, true);
        document.getElementById('viz-facet').innerHTML = interactiveVizFieldOptions(facetFields, true);
        document.getElementById('viz-animation').innerHTML = interactiveVizFieldOptions(animationFields, true);
        document.getElementById('viz-detail-fields').innerHTML = interactiveVizFieldOptions(fields, false);
        document.getElementById('viz-filter-field').innerHTML = interactiveVizFieldOptions(measures, true);
        document.getElementById('viz-category-filter-field').innerHTML = interactiveVizFieldOptions(categoryFields, true);
        configureInteractiveVizCapabilities(fields);

        const signature = fields.map(field => field.name).join('\u0001');
        if (previous && previousSignature === signature) {
            setInteractiveVizValue('viz-intent', previous.intent || 'custom');
            setInteractiveVizValue('viz-chart-type', previous.chart);
            ['x', 'y', 'color', 'size', 'facet', 'animation'].forEach(channel =>
                setInteractiveVizValue(`viz-${channel}`, previous[channel])
            );
            setInteractiveVizValue('viz-aggregation', previous.aggregation);
            setInteractiveVizValue('viz-time-unit', previous.timeUnit);
            const selected = new Set(previous.details);
            Array.from(document.getElementById('viz-detail-fields').options).forEach(option => {
                option.selected = selected.has(option.value);
            });
        } else {
            applyInteractiveVizRecommendation(false);
        }
        let validation = synchronizeInteractiveVizControls();
        if (validation.errors.length) {
            applyInteractiveVizRecommendation(false);
            validation = synchronizeInteractiveVizControls();
            setInteractiveVizConfigNote(['旧配置与当前字段语义不兼容，已恢复安全推荐'], false);
        }
        configureInteractiveVizRange();
        configureInteractiveVizCategories();
        const reason = interactiveVizSchema.recommendation?.reason || '已根据字段语义生成安全默认图。';
        const capability = interactiveVizSchema.capability?.summary || '';
        document.getElementById('interactive-viz-recommendation').textContent = `数据能力：${capability} 推荐依据：${reason}`;
        status.textContent = `已画像 ${fields.length} 个字段；${capability}`;
        await runInteractiveVisualization();
    } catch (error) {
        status.textContent = '交互图形工作台不可用：' + error.message;
    }
}

function applyInteractiveVizRecommendation(run = true) {
    if (!interactiveVizSchema) return;
    const recommendation = interactiveVizSchema.recommendation || {};
    const encodings = recommendation.encodings || {};
    setInteractiveVizValue('viz-intent', 'auto');
    syncInteractiveVizIntentButtons();
    setInteractiveVizValue('viz-chart-type', recommendation.chart_type || 'auto');
    ['x', 'y', 'color', 'size', 'facet', 'animation'].forEach(channel =>
        setInteractiveVizValue(`viz-${channel}`, encodings[channel] || '')
    );
    setInteractiveVizValue('viz-aggregation', recommendation.aggregation?.function || 'none');
    setInteractiveVizValue('viz-time-unit', recommendation.aggregation?.time_unit || 'none');
    document.getElementById('viz-filter-enabled').checked = false;
    Array.from(document.getElementById('viz-detail-fields').options).forEach(option => {
        option.selected = false;
    });
    const explanation = document.getElementById('interactive-viz-recommendation');
    const capability = interactiveVizSchema.capability?.summary || '';
    if (explanation) explanation.textContent = `数据能力：${capability} 推荐依据：${recommendation.reason || '按字段语义选择安全默认图。'}`;
    const validation = synchronizeInteractiveVizControls();
    if (run && !validation.errors.length) runInteractiveVisualization();
}

function resetInteractiveVisualization() {
    stopInteractiveVizPlayback();
    applyInteractiveVizRecommendation(false);
    configureInteractiveVizRange();
    configureInteractiveVizCategories();
    runInteractiveVisualization();
}

function configureInteractiveVizRange() {
    const field = interactiveVizField(document.getElementById('viz-filter-field')?.value);
    const lower = document.getElementById('viz-range-min');
    const upper = document.getElementById('viz-range-max');
    if (!field?.range || field.kind !== 'numeric') {
        lower.disabled = true;
        upper.disabled = true;
        document.getElementById('viz-range-min-value').textContent = '-';
        document.getElementById('viz-range-max-value').textContent = '-';
        return;
    }
    lower.disabled = false;
    upper.disabled = false;
    lower.value = 0;
    upper.value = 1000;
    syncInteractiveVizRange();
}

function interactiveVizRangeValue(field, sliderValue) {
    const minimum = Number(field.range[0]);
    const maximum = Number(field.range[1]);
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return null;
    return minimum + (maximum - minimum) * Number(sliderValue) / 1000;
}

function formatInteractiveVizNumber(value) {
    if (!Number.isFinite(Number(value))) return '-';
    const number = Number(value);
    const magnitude = Math.abs(number);
    if ((magnitude > 0 && magnitude < 0.001) || magnitude >= 1e6) return number.toExponential(3);
    return Number(number.toPrecision(6)).toLocaleString();
}

function syncInteractiveVizRange(changed = '') {
    const lower = document.getElementById('viz-range-min');
    const upper = document.getElementById('viz-range-max');
    if (Number(lower.value) > Number(upper.value)) {
        if (changed === 'min') upper.value = lower.value;
        else lower.value = upper.value;
    }
    const field = interactiveVizField(document.getElementById('viz-filter-field')?.value);
    if (field?.range) {
        document.getElementById('viz-range-min-value').textContent = formatInteractiveVizNumber(interactiveVizRangeValue(field, lower.value));
        document.getElementById('viz-range-max-value').textContent = formatInteractiveVizNumber(interactiveVizRangeValue(field, upper.value));
    }
    if (document.getElementById('viz-filter-enabled')?.checked) scheduleInteractiveVisualization();
}

function configureInteractiveVizCategories() {
    const field = interactiveVizField(document.getElementById('viz-category-filter-field')?.value);
    const values = document.getElementById('viz-category-filter-values');
    values.innerHTML = (field?.levels || []).map(level =>
        `<option value="${escapeHtml(String(level.value))}">${escapeHtml(String(level.value))} · ${Number(level.count_profiled || 0).toLocaleString()}</option>`
    ).join('');
    scheduleInteractiveVisualization();
}

function syncInteractiveVizSliders(requiresData) {
    document.getElementById('viz-bins-value').textContent = document.getElementById('viz-bins').value;
    document.getElementById('viz-max-points-value').textContent = Number(document.getElementById('viz-max-points').value).toLocaleString();
    document.getElementById('viz-opacity-value').textContent = document.getElementById('viz-opacity').value + '%';
    document.getElementById('viz-symbol-size-value').textContent = document.getElementById('viz-symbol-size').value + '%';
    if (requiresData) scheduleInteractiveVisualization();
    else renderInteractiveVisualization();
}

function scheduleInteractiveVisualization(delay = 350) {
    if (!interactiveVizSchema) return;
    const validation = synchronizeInteractiveVizControls();
    if (validation.errors.length) return;
    clearTimeout(interactiveVizRefreshTimer);
    interactiveVizRefreshTimer = setTimeout(() => runInteractiveVisualization(), delay);
}

function buildInteractiveVisualizationSpecification() {
    const chartType = document.getElementById('viz-chart-type').value;
    const detailSelect = document.getElementById('viz-detail-fields');
    let details = Array.from(detailSelect.selectedOptions).map(option => option.value);
    if (chartType === 'parallel' && details.length < 2) {
        const numeric = (interactiveVizSchema.fields || []).filter(field => field.kind === 'numeric').slice(0, 6).map(field => field.name);
        details = numeric;
        const selected = new Set(numeric);
        Array.from(detailSelect.options).forEach(option => { option.selected = selected.has(option.value); });
    }
    const encodings = {
        x: document.getElementById('viz-x').value || null,
        y: document.getElementById('viz-y').value || null,
        color: document.getElementById('viz-color').value || null,
        size: document.getElementById('viz-size').value || null,
        facet: document.getElementById('viz-facet').value || null,
        animation: document.getElementById('viz-animation').value || null,
        tooltip: details,
        parallel: details,
    };
    const filters = [];
    const rangeField = interactiveVizField(document.getElementById('viz-filter-field').value);
    if (rangeField?.range && document.getElementById('viz-filter-enabled').checked) {
        filters.push({
            field: rangeField.name,
            kind: 'range',
            min: interactiveVizRangeValue(rangeField, document.getElementById('viz-range-min').value),
            max: interactiveVizRangeValue(rangeField, document.getElementById('viz-range-max').value),
        });
    }
    const categoryField = document.getElementById('viz-category-filter-field').value;
    const categoryValues = Array.from(document.getElementById('viz-category-filter-values').selectedOptions).map(option => option.value);
    if (categoryField && categoryValues.length) {
        filters.push({ field: categoryField, kind: 'in', values: categoryValues });
    }
    return {
        chart_type: chartType,
        encodings,
        filters,
        aggregation: {
            function: document.getElementById('viz-aggregation').value,
            group_by: [],
            time_unit: document.getElementById('viz-time-unit').value,
            bins: Number(document.getElementById('viz-bins').value),
        },
        max_points: Number(document.getElementById('viz-max-points').value),
    };
}

async function runInteractiveVisualization() {
    if (!interactiveVizSchema) return;
    const status = document.getElementById('interactive-viz-status');
    const validation = synchronizeInteractiveVizControls();
    if (validation.errors.length) {
        status.classList.add('is-warning');
        status.textContent = '图形未生成：' + validation.errors.join('；');
        return;
    }
    const requestId = ++interactiveVizRequestId;
    status.classList.remove('is-warning');
    status.textContent = '正在编译筛选、粒度、聚合和有界图元…';
    try {
        const response = await fetch('/api/visualization/explore/data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(buildInteractiveVisualizationSpecification()),
        });
        const payload = await response.json();
        if (requestId !== interactiveVizRequestId) return;
        if (!payload.success) throw new Error(payload.error || '交互图编译失败');
        interactiveVizResult = payload.result;
        const facetLevels = interactiveVizResult.facet_levels || [];
        const facetSlider = document.getElementById('viz-facet-slider');
        facetSlider.max = Math.max(0, facetLevels.length - 4);
        facetSlider.value = Math.min(Number(facetSlider.value || 0), Number(facetSlider.max));
        document.getElementById('viz-facet-controls').classList.toggle('hidden', facetLevels.length <= 1);
        const animationLevels = interactiveVizResult.animation_levels || [];
        const animationSlider = document.getElementById('viz-animation-slider');
        animationSlider.max = animationLevels.length;
        animationSlider.value = Math.min(Number(animationSlider.value || 0), Number(animationSlider.max));
        document.getElementById('viz-animation-controls').classList.toggle('hidden', animationLevels.length === 0);
        renderInteractiveVisualization();
        const audit = interactiveVizResult.audit || {};
        const warnings = interactiveVizResult.warnings || [];
        status.classList.toggle('is-warning', warnings.length > 0);
        status.textContent = `${interactiveVizChartLabel(interactiveVizResult.chart_type)} · 筛选后 ${Number(audit.filtered_rows || 0).toLocaleString()} 行 · 浏览器图元 ${Number(audit.output_rows || 0).toLocaleString()} · ${Number(interactiveVizResult.timing_ms || 0).toFixed(1)} ms${warnings.length ? ` · ⚠ ${warnings[0]}` : ''}`;
    } catch (error) {
        status.classList.add('is-warning');
        status.textContent = '图形未更新：' + error.message;
    }
}

function interactiveVizTooltipFormatter(params, fields) {
    const item = Array.isArray(params) ? params[0] : params;
    const raw = item?.data?.raw || {};
    const lines = fields.filter((field, index) => field && fields.indexOf(field) === index).map(field =>
        `<strong>${escapeHtml(interactiveVizDisplayField(String(field)))}</strong>: ${escapeHtml(String(raw[field] ?? '缺失'))}`
    );
    return lines.join('<br>');
}

function groupInteractiveVizRecords(records, colorField, numericColor, maxGroups = 12) {
    if (!colorField || numericColor) return new Map([['数据', records]]);
    const rawGroups = new Map();
    records.forEach(record => {
        const key = String(record[colorField] ?? '缺失');
        if (!rawGroups.has(key)) rawGroups.set(key, []);
        rawGroups.get(key).push(record);
    });
    if (rawGroups.size <= maxGroups) return rawGroups;
    const ranked = Array.from(rawGroups.entries()).sort((left, right) =>
        right[1].length - left[1].length || left[0].localeCompare(right[0], 'zh-CN')
    );
    const kept = new Set(ranked.slice(0, maxGroups - 1).map(([name]) => name));
    const compact = new Map();
    ranked.forEach(([name, groupRecords]) => {
        const compactName = kept.has(name) ? name : '其他';
        if (!compact.has(compactName)) compact.set(compactName, []);
        compact.get(compactName).push(...groupRecords);
    });
    return compact;
}

function renderInteractiveVisualization() {
    if (!interactiveVizResult || typeof echarts === 'undefined') return;
    const chartDom = document.getElementById('interactive-viz-chart');
    if (!chartDom) return;
    interactiveVizChart = echarts.getInstanceByDom(chartDom) || echarts.init(chartDom);
    const result = interactiveVizResult;
    const enc = result.encodings || {};
    let records = result.records || [];
    const animationLevels = result.animation_levels || [];
    const animationIndex = Number(document.getElementById('viz-animation-slider').value || 0);
    if (enc.animation && animationIndex > 0 && animationLevels[animationIndex - 1] !== undefined) {
        const active = String(animationLevels[animationIndex - 1]);
        records = records.filter(record => String(record[enc.animation]) === active);
        document.getElementById('viz-animation-value').textContent = active;
    } else {
        document.getElementById('viz-animation-value').textContent = '全部';
    }
    const facetLevels = result.facet_levels || [];
    const facetStart = Number(document.getElementById('viz-facet-slider').value || 0);
    const visibleFacets = enc.facet ? facetLevels.slice(facetStart, facetStart + 4) : [null];
    document.getElementById('viz-facet-value').textContent = enc.facet && facetLevels.length
        ? `${facetStart + 1}–${Math.min(facetStart + visibleFacets.length, facetLevels.length)} / ${facetLevels.length}` : '-';
    const opacity = Number(document.getElementById('viz-opacity').value) / 100;
    const sizeScale = Number(document.getElementById('viz-symbol-size').value) / 100;
    const tooltipFields = [enc.x, enc.y, enc.color, enc.size, enc.facet, enc.animation, ...(enc.tooltip || [])];

    if (result.chart_type === 'parallel') {
        const axes = enc.parallel || [];
        const colorField = enc.color;
        const parallelNumericColor = colorField && result.field_types?.[colorField] === 'numeric';
        const rawGroups = groupInteractiveVizRecords(records, colorField, parallelNumericColor);
        const groups = new Map(Array.from(rawGroups.entries()).map(([name, groupRecords]) => [
            name, groupRecords.map(record => axes.map(field => record[field])),
        ]));
        const parallelAxis = axes.map((field, index) => {
            const kind = result.field_types?.[field];
            const axis = { dim: index, name: field };
            if (kind !== 'numeric') {
                axis.type = 'category';
                axis.data = Array.from(new Set(records.map(record => record[field]).filter(value => value != null))).slice(0, 50);
            }
            return axis;
        });
        chartDom.style.height = '580px';
        interactiveVizChart.clear();
        interactiveVizChart.setOption({
            animation: false,
            color: ['#2563eb', '#0f9f8f', '#f59e0b', '#e05260', '#7c63d6', '#3196b8', '#84a33d', '#c76aa4'],
            tooltip: { trigger: 'item', backgroundColor: 'rgba(12,30,49,.94)', borderWidth: 0, textStyle: { color: '#edf6fb', fontSize: 11 } },
            legend: { type: 'scroll', top: 10, textStyle: { color: '#53697e', fontSize: 10 } },
            toolbox: { right: 12, top: 8, iconStyle: { borderColor: '#7890a6' }, feature: { saveAsImage: {}, restore: {} } },
            parallel: { left: 65, right: 55, top: 72, bottom: 35, parallelAxisDefault: { type: 'value', nameLocation: 'end', nameTextStyle: { color: '#405a72', fontWeight: 600 }, axisLine: { lineStyle: { color: '#9cb0c2' } }, axisLabel: { color: '#667d91' }, splitLine: { lineStyle: { color: '#e7edf3' } } } },
            parallelAxis,
            series: Array.from(groups.entries()).map(([name, data]) => ({
                name, type: 'parallel', data, lineStyle: { width: 1.2, opacity }, progressive: 1000,
            })),
        }, true);
        interactiveVizChart.resize();
        renderInteractiveVizAudit();
        renderInteractiveVizInsight(records);
        return;
    }

    const facets = visibleFacets.length ? visibleFacets : [null];
    const columns = facets.length > 1 ? 2 : 1;
    const rows = Math.ceil(facets.length / columns);
    chartDom.style.height = `${Math.max(440, rows * 300 + 80)}px`;
    const grids = [];
    const xAxes = [];
    const yAxes = [];
    const titles = [];
    const series = [];
    const legendNames = new Set();
    const colorKind = enc.color ? result.field_types?.[enc.color] : null;
    const numericColor = colorKind === 'numeric';
    const xKind = result.field_types?.[enc.x];
    const yKind = result.field_types?.[enc.y];
    const sizeRange = result.numeric_ranges?.[enc.size] || [0, 1];
    const colorRange = result.numeric_ranges?.[enc.color] || [0, 1];
    const symbolSize = value => {
        if (!enc.size) return Math.max(3, 8 * sizeScale);
        const raw = Number(value[3]);
        const span = Number(sizeRange[1]) - Number(sizeRange[0]);
        const normalized = Number.isFinite(raw) && span > 0 ? (raw - Number(sizeRange[0])) / span : 0.4;
        return (5 + 24 * Math.max(0, Math.min(1, normalized))) * sizeScale;
    };

    facets.forEach((facetValue, facetIndex) => {
        const row = Math.floor(facetIndex / columns);
        const column = facetIndex % columns;
        const subset = enc.facet
            ? records.filter(record => String(record[enc.facet]) === String(facetValue))
            : records;
        grids.push({
            left: columns === 2 ? (column === 0 ? '7%' : '55%') : '8%',
            width: columns === 2 ? '38%' : '82%',
            top: 55 + row * 300,
            height: 215,
            containLabel: true,
        });
        xAxes.push({
            gridIndex: facetIndex,
            type: xKind === 'numeric' ? 'value' : (xKind === 'datetime' ? 'time' : 'category'),
            name: interactiveVizDisplayField(enc.x), nameLocation: 'middle', nameGap: 30,
            axisLabel: {
                hideOverlap: true,
                rotate: !['numeric', 'datetime'].includes(xKind) ? 25 : 0,
                color: '#667d91',
                fontSize: 10,
                formatter: xKind === 'datetime' ? value => {
                    const date = new Date(value);
                    return Number.isNaN(date.getTime()) ? String(value) : `${date.getMonth() + 1}/${date.getDate()}`;
                } : undefined,
            },
            axisLine: { lineStyle: { color: '#a7b8c8' } }, axisTick: { lineStyle: { color: '#a7b8c8' } },
            splitLine: { show: xKind === 'numeric', lineStyle: { color: '#e9eef4', type: 'dashed' } },
            nameTextStyle: { color: '#536b82', fontWeight: 600, fontSize: 11 },
            scale: true,
        });
        yAxes.push({
            gridIndex: facetIndex,
            type: yKind === 'numeric' ? 'value' : 'category',
            name: interactiveVizDisplayField(enc.y), nameLocation: 'middle', nameGap: 45,
            axisLabel: { hideOverlap: true, color: '#667d91', fontSize: 10 },
            axisLine: { lineStyle: { color: '#a7b8c8' } }, axisTick: { lineStyle: { color: '#a7b8c8' } },
            splitLine: { show: true, lineStyle: { color: '#e9eef4', type: 'dashed' } },
            nameTextStyle: { color: '#536b82', fontWeight: 600, fontSize: 11 }, scale: true,
        });
        if (enc.facet) {
            titles.push({
                text: `${enc.facet}: ${String(facetValue)}`,
                left: columns === 2 ? (column === 0 ? '7%' : '55%') : '8%',
                top: 28 + row * 300,
                textStyle: { fontSize: 13, fontWeight: 600 },
            });
        }
        const groups = groupInteractiveVizRecords(subset, enc.color, numericColor);
        Array.from(groups.entries()).forEach(([groupName, groupRecords]) => {
            legendNames.add(groupName);
            const data = groupRecords.map(record => ({
                value: [record[enc.x], record[enc.y], numericColor ? record[enc.color] : null, enc.size ? record[enc.size] : null],
                raw: record,
            }));
            series.push({
                name: groupName,
                type: result.chart_type === 'area' ? 'line' : result.chart_type,
                xAxisIndex: facetIndex,
                yAxisIndex: facetIndex,
                data,
                symbolSize,
                showSymbol: result.chart_type === 'line' || result.chart_type === 'area' ? data.length < 500 : true,
                areaStyle: result.chart_type === 'area' ? { opacity: Math.min(opacity, 0.45) } : undefined,
                lineStyle: { width: 2.1, opacity },
                itemStyle: result.chart_type === 'bar'
                    ? { opacity, borderRadius: [5, 5, 1, 1] }
                    : { opacity, borderColor: '#ffffff', borderWidth: result.chart_type === 'scatter' ? 1 : 0 },
                barMaxWidth: 46,
                large: result.chart_type === 'scatter' && data.length > 2000,
                largeThreshold: 2000,
                progressive: 2000,
                emphasis: { focus: 'series' },
            });
        });
    });
    const xAxisIndices = xAxes.map((_, index) => index);
    const yAxisIndices = yAxes.map((_, index) => index);
    const xLevelCount = enc.x ? new Set(records.map(record => String(record[enc.x]))).size : 0;
    const yLevelCount = enc.y ? new Set(records.map(record => String(record[enc.y]))).size : 0;
    const showXSlider = !['numeric', 'datetime'].includes(xKind)
        ? xLevelCount > 18 : records.length > 800;
    const showYSlider = yKind !== 'numeric'
        ? yLevelCount > 18 : records.length > 1200;
    const dataZoom = [
        { type: 'inside', xAxisIndex: xAxisIndices, filterMode: 'filter' },
        { type: 'inside', yAxisIndex: yAxisIndices, filterMode: 'empty' },
    ];
    if (showXSlider) {
        dataZoom.push({ type: 'slider', xAxisIndex: xAxisIndices, bottom: 8, height: 18, filterMode: 'filter' });
    }
    if (showYSlider) {
        dataZoom.push({ type: 'slider', yAxisIndex: yAxisIndices, right: 3, width: 16, filterMode: 'empty' });
    }
    const option = {
        animation: records.length < 2500,
        animationDuration: 420,
        color: ['#2563eb', '#0f9f8f', '#f59e0b', '#e05260', '#7c63d6', '#3196b8', '#84a33d', '#c76aa4'],
        textStyle: { fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif" },
        title: titles,
        legend: legendNames.size > 1 ? { type: 'scroll', top: 8, left: 'center', data: Array.from(legendNames), textStyle: { color: '#53697e', fontSize: 10 }, itemWidth: 14, itemHeight: 8 } : undefined,
        toolbox: { right: 12, top: 6, iconStyle: { borderColor: '#7890a6' }, emphasis: { iconStyle: { borderColor: '#2563eb' } }, feature: { saveAsImage: {}, dataZoom: {}, restore: {} } },
        tooltip: { trigger: 'item', confine: true, backgroundColor: 'rgba(12,30,49,.94)', borderWidth: 0, padding: [9, 11], textStyle: { color: '#edf6fb', fontSize: 11 }, extraCssText: 'box-shadow:0 10px 28px rgba(7,23,39,.22);border-radius:8px;', formatter: params => interactiveVizTooltipFormatter(params, tooltipFields) },
        grid: grids,
        xAxis: xAxes,
        yAxis: yAxes,
        dataZoom,
        series,
    };
    if (numericColor && Number.isFinite(Number(colorRange[0])) && Number.isFinite(Number(colorRange[1]))) {
        option.visualMap = {
            type: 'continuous', dimension: 2, min: Number(colorRange[0]), max: Number(colorRange[1]),
            calculable: true, orient: 'horizontal', left: 'center', bottom: 32,
            inRange: { color: ['#2563eb', '#22c55e', '#f59e0b', '#dc2626'] },
            seriesIndex: series.map((_, index) => index),
        };
    }
    interactiveVizChart.clear();
    interactiveVizChart.setOption(option, true);
    interactiveVizChart.resize();
    renderInteractiveVizAudit();
    renderInteractiveVizInsight(records);
}

function renderInteractiveVizInsight(visibleRecords = null) {
    const title = document.getElementById('interactive-viz-insight-title');
    const body = document.getElementById('interactive-viz-insight-body');
    const boundary = document.getElementById('interactive-viz-insight-boundary');
    if (!title || !body || !boundary || !interactiveVizResult) return;

    const result = interactiveVizResult;
    const records = visibleRecords || result.records || [];
    const enc = result.encodings || {};
    const audit = result.audit || {};
    const warnings = result.warnings || [];
    const aggregation = (audit.aggregation || {}).function || 'none';
    const xLabel = interactiveVizDisplayField(enc.x);
    const yLabel = interactiveVizDisplayField(enc.y);
    const countMode = enc.y === '__count__' || aggregation === 'count';

    if (result.chart_type === 'bar' && countMode) {
        const totals = new Map();
        records.forEach(record => {
            const key = String(record[enc.x] ?? '缺失');
            totals.set(key, (totals.get(key) || 0) + Number(record.__count__ || 0));
        });
        const ranked = Array.from(totals.entries()).sort((left, right) => right[1] - left[1]);
        const total = ranked.reduce((sum, item) => sum + item[1], 0);
        const top = ranked[0];
        title.textContent = `${xLabel}的数量构成`;
        body.textContent = top && total > 0
            ? `当前视图中“${top[0]}”数量最多，为 ${top[1].toLocaleString()} 条，占可见记录的 ${(top[1] / total * 100).toFixed(1)}%。`
            : '当前筛选范围内没有足够记录用于比较类别构成。';
        boundary.textContent = '数量差异只描述记录分布；若采样单位、缺失机制或收集频率不同，不能直接解释为真实规模差异。';
    } else if (result.chart_type === 'bar') {
        title.textContent = `${xLabel}之间的${yLabel}比较`;
        body.textContent = `当前按“${xLabel}”比较“${yLabel}”，使用${({ mean: '均值', sum: '总量', median: '中位数', min: '最小值', max: '最大值' })[aggregation] || '逐行'}口径，共显示 ${records.length.toLocaleString()} 个图元。`;
        boundary.textContent = '组间高低不等于统计显著；还需检查组内波动、样本量不平衡与异常值敏感性。';
    } else if (result.chart_type === 'scatter') {
        title.textContent = `${xLabel}与${yLabel}的联合分布`;
        body.textContent = `当前展示 ${records.length.toLocaleString()} 个可见观测，用于检查方向、非线性、分群与异常点；颜色分层${enc.color ? `使用“${enc.color}”` : '未启用'}。`;
        boundary.textContent = '散点形态只能提示关联，不证明因果；确认结论前应补做效应量、置信区间、混杂控制与样本外验证。';
    } else if (['line', 'area'].includes(result.chart_type)) {
        title.textContent = `${yLabel}随${xLabel}的变化轨迹`;
        body.textContent = `当前以“${xLabel}”排序观察“${yLabel}”，共显示 ${records.length.toLocaleString()} 个时间/有序位置${enc.color ? `，并按“${enc.color}”分层` : ''}。`;
        boundary.textContent = '趋势可能由季节性、结构突变或聚合粒度造成；预测前需检查时序连续性、滞后关系和末段外推稳定性。';
    } else if (result.chart_type === 'parallel') {
        title.textContent = '多变量轮廓与群体差异';
        body.textContent = `当前并行比较 ${(enc.parallel || []).length} 个字段、${records.length.toLocaleString()} 条可见轮廓，适合发现共同变化与异常组合。`;
        boundary.textContent = '轴的量纲与排列会明显影响视觉判断；应配合标准化、稳健距离和扰动稳定性验证。';
    } else {
        title.textContent = '当前探索视图';
        body.textContent = `已生成 ${interactiveVizChartLabel(result.chart_type)}，包含 ${records.length.toLocaleString()} 个可见图元。`;
        boundary.textContent = '图形用于提出和筛选假设，正式结论仍需可执行检验、反证与样本外验证。';
    }
    if (warnings.length) {
        body.textContent += ` 注意：${warnings[0]}`;
    }
}

function renderInteractiveVizAudit() {
    const box = document.getElementById('interactive-viz-audit');
    if (!box || !interactiveVizResult) return;
    const audit = interactiveVizResult.audit || {};
    const warnings = interactiveVizResult.warnings || [];
    const scopeLabels = { full_filtered_data: '完整筛选数据', coverage_sample: '覆盖样本' };
    const aggregationLabels = { none: '逐行', count: '计数', sum: '求和', mean: '均值', median: '中位数', min: '最小值', max: '最大值' };
    const aggregation = (audit.aggregation || {}).function || 'none';
    box.innerHTML = [
        `源数据 ${Number(audit.source_rows || 0).toLocaleString()} 行`,
        `筛选后 ${Number(audit.filtered_rows || 0).toLocaleString()} 行`,
        `扫描 ${Number(audit.scanned_rows || 0).toLocaleString()} 行`,
        `输出 ${Number(audit.output_rows || 0).toLocaleString()} 图元`,
        `范围 ${scopeLabels[audit.scan_scope] || audit.scan_scope || '-'}`,
        `聚合 ${aggregationLabels[aggregation] || aggregation}`,
        ...warnings,
    ].map(text => `<span>${escapeHtml(String(text))}</span>`).join('');
}

async function sendInteractiveVizToReport() {
    if (!interactiveVizResult) {
        showToast('请先生成一张可验证的探索图', 'error');
        return;
    }
    const result = interactiveVizResult;
    if (result.chart_type === 'parallel') {
        showToast('平行坐标是探索视图，静态报表暂不支持等价编排；请先固化为比较图或关系图', 'error');
        return;
    }
    const enc = result.encodings || {};
    if (!enc.x || !enc.y) {
        showToast('当前发现缺少可交付的轴字段', 'error');
        return;
    }
    const specification = buildInteractiveVisualizationSpecification();
    const aggregation = (result.audit?.aggregation || {}).function || specification.aggregation.function || 'none';
    const groupField = enc.color && interactiveVizField(enc.color)?.semantic_role !== 'measure' ? enc.color : '';
    const title = document.getElementById('interactive-viz-insight-title')?.textContent
        || `${interactiveVizDisplayField(enc.y)} / ${interactiveVizDisplayField(enc.x)}`;
    const chart = {
        id: nextChartId++,
        chart_type: result.chart_type,
        x_field: enc.x,
        y_field: enc.y,
        group_field: groupField,
        agg: aggregation,
        title,
        color_scheme: 'default',
        show_values: result.chart_type === 'bar',
        top_n: 0,
        filters: specification.filters || [],
        time_unit: specification.aggregation.time_unit || 'none',
        bins: specification.aggregation.bins || 20,
        discovery_note: document.getElementById('interactive-viz-insight-boundary')?.textContent || '',
    };

    chartConfigs.push(chart);
    goStep(3);
    await loadReportFields();
    renderChartList();
    const target = document.getElementById(`chart-config-${chart.id}`);
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    showToast('已加入报表编排台，并保留筛选、粒度和聚合口径');
}

function toggleInteractiveVizPlayback() {
    if (interactiveVizPlaybackTimer) {
        stopInteractiveVizPlayback();
        return;
    }
    const slider = document.getElementById('viz-animation-slider');
    if (Number(slider.max) < 1) return;
    document.getElementById('viz-animation-play').textContent = '⏸ 暂停';
    if (Number(slider.value) === 0) slider.value = 1;
    interactiveVizPlaybackTimer = setInterval(() => {
        let next = Number(slider.value) + 1;
        if (next > Number(slider.max)) next = 1;
        slider.value = next;
        renderInteractiveVisualization();
    }, 900);
}

function stopInteractiveVizPlayback() {
    if (interactiveVizPlaybackTimer) clearInterval(interactiveVizPlaybackTimer);
    interactiveVizPlaybackTimer = null;
    const button = document.getElementById('viz-animation-play');
    if (button) button.textContent = '▶ 播放';
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
        llmImageAttachments = [];
        researchImageAttachments = [];
        location.reload();
    }
}

// ==================== LLM 智能分析 ====================

function initLLMStep() {
    // 检查可用性并禁用不可用的分析类型
    checkLLMAnalysisAvailability();
    // 加载默认配置
    loadLLMDefaults();
    initLLMImageUpload();
}

const LLM_IMAGE_MAX_COUNT = 5;
const LLM_IMAGE_MAX_BYTES = 6 * 1024 * 1024;
const LLM_IMAGE_TOTAL_MAX_BYTES = 20 * 1024 * 1024;

function initLLMImageUpload() {
    const input = document.getElementById('llm-image-input');
    if (!input || input.dataset.bound === '1') return;
    input.dataset.bound = '1';
    input.addEventListener('change', async event => {
        const files = Array.from(event.target.files || []);
        event.target.value = '';
        if (!files.length) return;
        const remaining = LLM_IMAGE_MAX_COUNT - llmImageAttachments.length;
        if (remaining <= 0) {
            showToast(`最多上传 ${LLM_IMAGE_MAX_COUNT} 张图片`, 'error');
            return;
        }
        const accepted = files.slice(0, remaining);
        if (files.length > remaining) showToast(`仅添加前 ${remaining} 张图片`, 'info');
        for (const file of accepted) {
            if (!/^image\/(png|jpe?g|webp|gif)$/i.test(file.type)) {
                showToast(`${file.name} 不是支持的图片格式`, 'error');
                continue;
            }
            if (file.size <= 0 || file.size > LLM_IMAGE_MAX_BYTES) {
                showToast(`${file.name} 超过 6 MB 限制`, 'error');
                continue;
            }
            const currentTotal = llmImageAttachments.reduce((sum, image) => sum + image.size, 0);
            if (currentTotal + file.size > LLM_IMAGE_TOTAL_MAX_BYTES) {
                showToast('图片总大小不能超过 20 MB', 'error');
                break;
            }
            try {
                const dataUrl = await readFileAsDataUrl(file);
                llmImageAttachments.push({ name: file.name, mime_type: file.type.toLowerCase(), size: file.size, data_url: dataUrl });
            } catch (error) {
                showToast(`读取 ${file.name} 失败`, 'error');
            }
        }
        renderLLMImagePreview();
    });
    renderLLMImagePreview();
}

function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ''));
        reader.onerror = () => reject(reader.error || new Error('读取文件失败'));
        reader.readAsDataURL(file);
    });
}

function removeLLMImage(index) {
    llmImageAttachments.splice(index, 1);
    renderLLMImagePreview();
}

function renderLLMImagePreview() {
    const container = document.getElementById('llm-image-preview');
    const status = document.getElementById('llm-image-status');
    if (!container || !status) return;
    container.innerHTML = '';
    if (!llmImageAttachments.length) {
        status.textContent = '未选择图片（最多 5 张，每张不超过 6 MB）';
        return;
    }
    const total = llmImageAttachments.reduce((sum, image) => sum + image.size, 0);
    status.textContent = `已选择 ${llmImageAttachments.length}/${LLM_IMAGE_MAX_COUNT} 张 · ${(total / 1024 / 1024).toFixed(1)} MB`;
    llmImageAttachments.forEach((image, index) => {
        const card = document.createElement('div');
        card.className = 'llm-image-card';
        const preview = document.createElement('img');
        preview.className = 'llm-image-thumb';
        preview.src = image.data_url;
        preview.alt = image.name;
        const meta = document.createElement('div');
        meta.className = 'llm-image-meta';
        const name = document.createElement('span');
        name.textContent = image.name;
        const size = document.createElement('small');
        size.textContent = `${(image.size / 1024).toFixed(0)} KB`;
        meta.append(name, size);
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'llm-image-remove';
        remove.title = '移除图片';
        remove.textContent = '×';
        remove.addEventListener('click', () => removeLLMImage(index));
        card.append(preview, meta, remove);
        container.appendChild(card);
    });
}

async function loadLLMDefaults() {
    try {
        const res = await fetch('/api/llm/config');
        const data = await res.json();
        if (data.success && data.providers) {
            window.llmProviders = data.providers;
            onLLMProviderChange();
        }
    } catch (e) {
        console.error('加载 LLM 配置失败', e);
    }
}

function onLLMProviderChange() {
    const provider = document.getElementById('llm-provider').value;
    const providers = window.llmProviders || {};
    const fallback = provider === 'deepseek'
        ? { base_url: 'https://api.deepseek.com', model_name: 'deepseek-v4-pro', needs_api_key: true }
        : null;
    const cfg = providers[provider] || fallback;
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
    populateModelPresetSelect('llm-model-preset', provider, document.getElementById('llm-model-name').value);
    const status = document.getElementById('llm-test-status');
    if (status) status.textContent = '';
}

function useLLMModelPreset(value) {
    if (!value) return;
    const input = document.getElementById('llm-model-name');
    if (input) input.value = value;
}

function testLLMConnection() {
    return requestLLMConnectionTest({
        provider: document.getElementById('llm-provider').value,
        base_url: document.getElementById('llm-base-url').value.trim(),
        model_name: document.getElementById('llm-model-name').value.trim(),
        api_key: document.getElementById('llm-api-key').value,
    }, document.getElementById('llm-test-btn'), document.getElementById('llm-test-status'));
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

    if (provider === 'ollama' && llmImageAttachments.length) {
        showToast('图片分析请切换到 DeepSeek Vision 或其他支持 image_url 的模型', 'error');
        return;
    }

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
                images: llmImageAttachments.map(image => ({
                    name: image.name,
                    mime_type: image.mime_type,
                    data_url: image.data_url,
                })),
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
        filters: [],
        time_unit: 'none',
        bins: 20,
        discovery_note: '',
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
        {value: 'none', label: '不聚合（散点）'},
        {value: 'sum', label: '求和'},
        {value: 'mean', label: '平均'},
        {value: 'count', label: '计数'},
        {value: 'max', label: '最大'},
        {value: 'min', label: '最小'},
        {value: 'median', label: '中位数'},
    ];
    const colorOptions = [
        {value: 'default', label: '默认'},
        {value: 'pastel', label: '柔和'},
        {value: 'dark', label: '深色'},
        {value: 'bright', label: '明亮'},
    ];
    
    const fieldOptions = reportFields.map(f => `<option value="${escapeHtml(f.name)}">${escapeHtml(f.name)}</option>`).join('');
    const valueFieldOptions = `<option value="__count__">记录数（探索口径）</option>${fieldOptions}`;
    
    container.innerHTML = chartConfigs.map(chart => `
        <div class="chart-config-item" id="chart-config-${chart.id}">
            <div class="chart-config-header">
                <span class="chart-config-title">${escapeHtml(chart.title)}</span>
                <div style="display:flex;gap:6px;">
                    <button class="btn btn-sm" onclick="previewSingleChart(${chart.id})">👁 预览</button>
                    <button class="btn btn-sm" style="color:var(--danger);" onclick="removeChart(${chart.id})">🗑️ 删除</button>
                </div>
            </div>
            ${chart.discovery_note ? `<div class="chart-discovery-origin"><span>来自探索</span>${escapeHtml(chart.discovery_note)}</div>` : ''}
            <div class="chart-config-body">
                <div class="form-group">
                    <label>图表类型</label>
                    <select onchange="updateChart(${chart.id}, 'chart_type', this.value)">
                        ${chartTypes.map(t => `<option value="${t.value}" ${t.value === chart.chart_type ? 'selected' : ''}>${t.label}</option>`).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>标题</label>
                    <input type="text" value="${escapeHtml(chart.title)}" onchange="updateChart(${chart.id}, 'title', this.value)">
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
                        ${valueFieldOptions}
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
        filters: c.filters || [],
        time_unit: c.time_unit || 'none',
        bins: c.bins || 20,
        discovery_note: c.discovery_note || '',
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
