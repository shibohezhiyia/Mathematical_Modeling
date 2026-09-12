/* Explicit development -> optional bounded mutation -> frozen model -> final audit. */
function graphLabEscape(value) {
    return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function graphLabNumber(value) {
    return typeof value === 'number' && Number.isFinite(value) ? Number(value.toPrecision(6)).toString() : value ?? '—';
}
function graphLabFitLabel(fit) {
    const labels = {not_required:'无需拟合参数', local_fit_completed:'局部拟合完成（非全局最优证明）',
        bounded_linear_fit_completed:'结构证明为仿射 · 有界线性拟合完成',
        variable_projection_completed:'部分线性参数已消元 · 外层非线性搜索完成',
        fit_budget_exhausted:'拟合预算不足，未判定结构错误', numeric_failure:'拟合出现数值定义域错误',
        not_converged:'参数拟合未收敛'};
    const count = Array.isArray(fit?.attempts) ? fit.attempts.length : 0;
    return (Object.hasOwn(labels, fit?.status) ? labels[fit.status] : '无拟合记录') +
        (count ? ` · 尝试 ${count} 个起点` : '');
}
function renderGraphLab(state) {
    const esc = graphLabEscape;
    const stages = {running:'正在开发搜索', ready:'候选已冻结 · 等待留出验证', confirming:'只读留出验证中',
        done:'验证流程已结束', cancelled:'已取消 · 不产生完整结论', no_candidate:'没有候选通过开发检查', error:'运行未完成'};
    const final = state.confirmation;
    const verdict = final?.status === 'passed_finite_heldout_checks' ? '有限留出检查通过（不是数学证明）' :
        final?.status === 'failed_heldout_checks' ? '留出检查未通过' : final ? '留出计算未完成，不能判定模型错误' : '尚未验证';
    const budget = state.budget || {};
    const mutation = state.model_mutation || {};
    let output = `<div class="graph-lab-phase" role="status">${esc(stages[state.status] || '状态未知')}</div>`;
    output += '<div class="graph-lab-stages"><section><h4>01 · 开发搜索</h4><p>训练、结构变异与反例重放</p>';
    output += `<strong>${esc(state.candidate_count || 0)} 个可选候选</strong><p>评估 ${esc(budget.evaluations_charged ?? '—')} 次 · 耗时 ${esc(budget.elapsed_seconds ?? '—')} 秒</p></section>`;
    output += `<section><h4>02 · 冻结快照</h4><p>结构、参数与阈值不再调整</p><strong>${state.selected_hash ? '已固定一个候选' : '尚未形成'}</strong>`;
    if (state.selected_hash) output += `<code>${esc(state.selected_hash)}</code>`;
    output += `</section><section><h4>03 · 留出审计</h4><p>结果不回传搜索</p><strong>${esc(verdict)}</strong></section></div>`;
    if (mutation.enabled) {
        const accepted = (mutation.events || []).reduce((total, event) => total + (event.accepted_count || 0), 0);
        output += `<p class="hint">受限模型变异调用 ${esc(mutation.calls || 0)} 次，校验后接收 ${esc(accepted)} 个 patch；模型不参与裁决。</p>`;
    }
    if (state.cancel_requested) output += '<p>取消已请求，等待受控计算退出；已消费的留出数据不会被重置。</p>';
    if (state.error) output += `<p class="graph-lab-alert">${esc(state.error)}：运行未完成，请查看已有产物。不要把执行失败当作数学反例。</p>`;
    if (final) {
        output += `<p>已执行 ${esc(final.check_count ?? '—')} 项检查；失败 ${esc(final.failed_check_count ?? '—')} 项。样本外独立性与现实语义未证明。</p>`;
        if (final.metrics?.length) {
            output += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>输出</th><th>留出 RMSE</th><th>训练均值基线 RMSE</th></tr></thead><tbody>';
            for (const m of final.metrics) output += `<tr><td>${esc(m.output)}</td><td>${esc(graphLabNumber(m.rmse))}</td><td>${esc(graphLabNumber(m.training_mean_baseline_rmse))}</td></tr>`;
            output += '</tbody></table></div>';
        }
    }
    if (state.reports?.length) {
        const labels = {eligible_for_confirmation:'有限开发检查通过', rejected_by_checks:'开发检查未通过',
            rejected_by_replay:'旧反例复检未通过', execution_incomplete:'计算未完成', not_executable:'图不可执行', not_assessed:'未评估'};
        output += '<details><summary>查看候选执行记录</summary><div class="table-wrapper"><table class="data-table"><thead><tr><th>候选</th><th>开发状态</th><th>参数拟合</th><th>选参 RMSE</th></tr></thead><tbody>';
        for (const r of state.reports) output += `<tr><td>${esc(String(r.hypothesis_hash || '').slice(0, 16))}</td><td>${esc(labels[r.status] || '待检查')}</td><td>${esc(graphLabFitLabel(r.fit))}</td><td>${esc(graphLabNumber(r.search_rmse))}</td></tr>`;
        output += '</tbody></table></div></details>';
    }
    const files = {report:'论证报告', development:'开发证据', frozen:'冻结模型', confirmation:'留出证据', 'heldout-mapping':'验证表映射', manifest:'产物清单'};
    output += '<div class="graph-lab-downloads">';
    for (const name of state.downloads || []) {
        if (Object.hasOwn(files, name) && /^[a-f0-9]{32}$/.test(state.id || ''))
            output += `<a class="btn btn-sm" href="/api/graph-lab/${state.id}/artifact/${name}">${files[name]}</a>`;
    }
    return output + '</div>';
}

const graphLabUI = {id: null, state: null, bundle: null, busy: false, timer: null};
function graphLabControls() {
    const active = graphLabUI.busy || ['running', 'confirming'].includes(graphLabUI.state?.status);
    document.getElementById('graph-lab-start').disabled = active;
    document.getElementById('graph-lab-example').disabled = active;
    document.getElementById('graph-lab-input').disabled = active;
    document.getElementById('graph-lab-study').disabled = active;
    document.getElementById('graph-lab-model-mutation').disabled = active;
    document.getElementById('graph-lab-cancel').disabled = graphLabUI.busy || !['running', 'confirming'].includes(graphLabUI.state?.status);
    document.getElementById('graph-lab-confirm').disabled = active || graphLabUI.state?.status !== 'ready';
    document.getElementById('graph-lab-heldout').disabled = active || graphLabUI.state?.status !== 'ready';
    document.querySelectorAll('#graph-lab-table-builder input, #graph-lab-table-builder select, #graph-lab-table-builder textarea, #graph-lab-table-builder button').forEach(el => { el.disabled = active; });
    document.querySelectorAll('#graph-lab-heldout-table input, #graph-lab-heldout-table select, #graph-lab-heldout-table button').forEach(el => {
        el.disabled = active || graphLabUI.state?.status !== 'ready';
    });
}
function graphLabMessage(text) { document.getElementById('graph-lab-message').textContent = text; }
async function graphLabRequest(url, payload) {
    const response = await fetch(url, payload === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}
async function graphLabReadFile(id) {
    const file = document.getElementById(id).files[0];
    if (!file) throw new Error('请先选择 JSON 文件');
    if (file.size > 256000) throw new Error('文件不得超过 256 KB');
    return JSON.parse(await file.text());
}
async function graphLabExample() {
    if (graphLabUI.busy) return;
    graphLabUI.busy = true; graphLabControls();
    try {
        graphLabUI.bundle = await graphLabRequest('/api/graph-lab/example');
        graphLabUI.source = 'example';
        graphLabDetach();
        document.getElementById('graph-lab-input').value = '';
        graphLabMessage('已载入合成开发示例：仅用于学习流程，不代表真实题目效果。填写研究标识后点击运行。');
    } catch (error) { graphLabMessage(error.message); }
    finally { graphLabUI.busy = false; graphLabControls(); }
}
async function graphLabRefresh() {
    clearTimeout(graphLabUI.timer);
    if (!graphLabUI.id) return;
    const id = graphLabUI.id;
    try {
        const state = await graphLabRequest(`/api/graph-lab/${id}`);
        if (id !== graphLabUI.id) return;
        graphLabUI.state = state;
        if (state.status === 'done') graphLabMessage('审计已结束。查看有限检查结论、适用边界与下载证据；本次模型和参数未再调整。');
        document.getElementById('graph-lab-output').innerHTML = renderGraphLab(state);
        graphLabControls();
        if (['running', 'confirming'].includes(state.status)) graphLabUI.timer = setTimeout(graphLabRefresh, 1500);
    } catch (error) { if (id === graphLabUI.id) graphLabMessage(`${error.message}；可点击刷新状态。`); }
}
async function graphLabStart() {
    if (graphLabUI.busy) return;
    graphLabUI.busy = true; graphLabControls();
    try {
        const file = document.getElementById('graph-lab-input').files[0];
        const bundle = file ? await graphLabReadFile('graph-lab-input') : graphLabUI.bundle;
        if (!bundle) throw new Error('请选择开发实验文件，或载入合成示例');
        const study = document.getElementById('graph-lab-study').value.trim();
        if (!/^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(study)) throw new Error('研究标识须以英文字母开头，使用字母、数字、下划线或连字符，最多 64 字符');
        const useModelMutation = document.getElementById('graph-lab-model-mutation').checked;
        const mutationModel = useModelMutation ? {
            provider: document.getElementById('research-semantic-provider').value,
            base_url: document.getElementById('research-semantic-base-url').value.trim(),
            model_name: document.getElementById('research-semantic-model-name').value.trim(),
            api_key: document.getElementById('research-semantic-api-key').value,
            timeout_seconds: 15,
        } : null;
        const request = {bundle, study};
        if (mutationModel) request.mutation_model = mutationModel;
        const result = await graphLabRequest('/api/graph-lab/run', request);
        clearTimeout(graphLabUI.timer);
        graphLabUI.id = result.id;
        graphLabClearHeldoutMapping();
        graphLabUI.state = {status:'running'};
        document.getElementById('graph-lab-heldout').value = '';
        try { sessionStorage.setItem('graphLabRun', result.id); } catch (_) { /* optional recovery */ }
        graphLabMessage(useModelMutation
            ? '开发搜索已启动。模型只提出受限 patch，数值评价器独立裁决；此阶段不接收留出数据。'
            : '开发搜索已启动。此阶段不接收留出数据，也不调用模型 API。');
        await graphLabRefresh();
    } catch (error) { graphLabMessage(error.message); }
    finally { graphLabUI.busy = false; graphLabControls(); }
}
async function graphLabConfirm() {
    if (graphLabUI.busy || graphLabUI.state?.status !== 'ready') return;
    graphLabUI.busy = true; graphLabControls();
    try {
        const payload = await graphLabReadFile('graph-lab-heldout');
        await graphLabRequest(`/api/graph-lab/${graphLabUI.id}/confirm`, payload);
        graphLabUI.state.status = 'confirming';
        graphLabMessage('只读留出审计已启动。失败或取消不会释放已使用的数据。');
        await graphLabRefresh();
    } catch (error) { graphLabMessage(error.message); }
    finally { graphLabUI.busy = false; graphLabControls(); }
}
async function graphLabCancel() {
    try {
        await graphLabRequest(`/api/graph-lab/${graphLabUI.id}/cancel`, {});
        graphLabMessage('已请求取消，等待计算退出。');
        await graphLabRefresh();
    } catch (error) { graphLabMessage(error.message); }
}
if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', () => {
    if (!document.getElementById('graph-lab-output')) return;
    try {
        const id = sessionStorage.getItem('graphLabRun');
        if (/^[a-f0-9]{32}$/.test(id || '')) graphLabUI.id = id;
    } catch (_) { /* storage is optional */ }
    graphLabControls(); graphLabRefresh();
});

function graphLabDetach() {
    clearTimeout(graphLabUI.timer);
    graphLabUI.id = null; graphLabUI.state = null;
    document.getElementById('graph-lab-output').innerHTML = '';
    document.getElementById('graph-lab-heldout').value = '';
    graphLabClearHeldoutMapping();
    try { sessionStorage.removeItem('graphLabRun'); } catch (_) { /* optional */ }
    graphLabControls();
}
async function graphLabTableFields() {
    if (graphLabUI.busy) return;
    graphLabUI.busy = true; graphLabControls();
    try {
        const fields = await graphLabRequest('/api/graph-lab/table-fields');
        graphLabUI.tableFields = fields;
        const options = fields.columns.filter(c => !c.identifier).map(c => `<option value="${graphLabEscape(c.name)}">${graphLabEscape(c.name)}</option>`).join('');
        document.getElementById('graph-lab-table-inputs').innerHTML = options;
        document.getElementById('graph-lab-table-target').innerHTML = '<option value="">请选择目标</option>' + options;
        document.getElementById('graph-lab-table-count').value = Math.min(200, fields.rows);
        document.getElementById('graph-lab-table-source').textContent = `当前表格 ${fields.rows} 行；疑似编号已排除。选择范围不是全表分析，也未验证行顺序的时间/实体独立性。`;
        graphLabTableUnits();
        graphLabInvalidateTable();
    } catch (error) { graphLabMessage(error.message === 'no_current_table' ? '请先在上方上传或处理表格。' : error.message); }
    finally { graphLabUI.busy = false; graphLabControls(); }
}
function graphLabTableUnits() {
    const current = new Map([...document.querySelectorAll('#graph-lab-table-units select')].map(el => [el.dataset.column, el.value]));
    const inputs = [...document.getElementById('graph-lab-table-inputs').selectedOptions].map(o => o.value);
    const target = document.getElementById('graph-lab-table-target').value;
    const names = [...new Set([...inputs, target].filter(Boolean))];
    const units = graphLabUI.tableFields?.units || [];
    document.getElementById('graph-lab-table-units').innerHTML = names.map(name => `<label>${graphLabEscape(name)} 的单位<select data-column="${graphLabEscape(name)}"><option value="">必须明确选择</option>${units.map(unit => `<option value="${graphLabEscape(unit)}"${current.get(name) === unit ? ' selected' : ''}>${graphLabEscape(unit === '1' ? '1（无量纲）' : unit)}</option>`).join('')}</select></label>`).join('');
}
function graphLabInvalidateTable() {
    if (graphLabUI.source !== 'table') return;
    graphLabUI.bundle = null;
    graphLabDetach();
    document.getElementById('graph-lab-table-preview').textContent = '绑定设置已改变，请重新准备实验；旧证据仍保存在本地运行目录。';
}
async function graphLabCompileTable() {
    if (graphLabUI.busy) return;
    graphLabUI.busy = true; graphLabControls();
    try {
        const config = {problem: document.getElementById('graph-lab-table-problem').value.trim(),
            inputs: [...document.getElementById('graph-lab-table-inputs').selectedOptions].map(o => o.value),
            target: document.getElementById('graph-lab-table-target').value,
            units: Object.fromEntries([...document.querySelectorAll('#graph-lab-table-units select')].map(el => [el.dataset.column, el.value])),
            start_row: Number(document.getElementById('graph-lab-table-start').value) - 1,
            row_count: Number(document.getElementById('graph-lab-table-count').value),
            absolute_tolerance: Number(document.getElementById('graph-lab-table-atol').value),
            relative_tolerance: Number(document.getElementById('graph-lab-table-rtol').value)};
        const result = await graphLabRequest('/api/graph-lab/compile-table', config);
        graphLabUI.bundle = result.bundle; graphLabUI.source = 'table';
        document.getElementById('graph-lab-input').value = '';
        graphLabDetach();
        const b = result.binding, esc = graphLabEscape;
        document.getElementById('graph-lab-table-preview').innerHTML = `<strong>实验已准备，尚未运行</strong><p>来源 ${b.source_rows} 行，本次第 ${b.start_row+1}–${b.start_row+b.selected_rows} 行：训练 ${b.training_rows} 行，开发检查 ${b.search_rows} 行。</p><p>${b.inputs.map(i => `${esc(i.node)} ← ${esc(i.column)}（${esc(i.unit)}，SI 换算 ×${esc(graphLabNumber(i.si_factor))}）`).join('；')}<br>y ← ${esc(b.output.column)}（${esc(b.output.unit)}，SI 换算 ×${esc(graphLabNumber(b.output.si_factor))}）</p><p>使用仿射原语组合作为搜索起点，参数搜索范围仅由训练数据尺度确定，不是题目物理边界。${b.full_table_analyzed ? '当前范围覆盖全表。' : '仅分析所选范围，不代表全表。'}</p>`;
        graphLabMessage('已自动生成带量纲的数学图和开发实验。核对绑定后，填写研究标识并点击“运行通用搜索”。');
    } catch (error) {
        const errors = {explicit_supported_units_required:'请为每个变量明确选择受支持的单位。', distinct_columns_required:'输入列与目标列必须不同，输入不能重复。',
            repeated_input_requires_grouped_design:'存在完全相同的输入。需要分组/重复观测设计，当前不会擅自去重。',
            missing_or_nonfinite_table_values:'所选范围有缺失或非有限数值，请先明确处理策略。', invalid_development_window:'请明确选择表内 12–360 行开发数据。',
            input_columns_required:'请选择 1–4 个输入列。', numeric_columns_required:'请选择当前表格中存在的数值列。',
            problem_text_required:'请填写题目背景或研究目标。', constant_training_input:'训练部分有常量输入，当前不能识别其系数。'};
        graphLabMessage(errors[error.message] || error.message);
    } finally { graphLabUI.busy = false; graphLabControls(); }
}
if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', () => {
    const builder = document.getElementById('graph-lab-table-builder');
    if (builder) {
        builder.addEventListener('input', graphLabInvalidateTable);
        builder.addEventListener('change', graphLabInvalidateTable);
    }
    document.getElementById('graph-lab-input')?.addEventListener('change', () => {
        graphLabUI.source = 'file'; graphLabUI.bundle = null; graphLabDetach();
    });
});

function graphLabClearHeldoutMapping() {
    graphLabUI.heldoutContext = null;
    document.getElementById('graph-lab-heldout-mapping').innerHTML = '';
    document.getElementById('graph-lab-heldout-source').textContent = '';
    document.getElementById('graph-lab-heldout-si').checked = false;
}
async function graphLabHeldoutFields() {
    if (graphLabUI.busy || graphLabUI.state?.status !== 'ready') return;
    graphLabUI.busy = true; graphLabControls();
    const id = graphLabUI.id;
    try {
        const start = Number(document.getElementById('graph-lab-heldout-start').value)-1;
        const fields = await graphLabRequest(`/api/graph-lab/${id}/heldout-fields?start=${encodeURIComponent(start)}`);
        if (id !== graphLabUI.id) return;
        graphLabClearHeldoutMapping();
        graphLabUI.heldoutContext = {id, hash: fields.frozen_model_hash, token: fields.table_snapshot};
        const esc = graphLabEscape;
        const columns = fields.columns.filter(c => !c.identifier);
        document.getElementById('graph-lab-heldout-mapping').innerHTML = [...fields.inputs.map(r => ({...r, kind:'输入'})),
            ...fields.outputs.map(r => ({...r, kind:'目标'}))].map(r => `<div class="graph-lab-fields"><label>${esc(r.kind)} ${esc(r.node)} · ${esc(r.name)}<select data-node="${esc(r.node)}" data-role="column"><option value="">请选择对应列</option>${columns.map(c => `<option value="${esc(c.name)}"${c.name === r.name ? ' selected' : ''}>${esc(c.name)}</option>`).join('')}</select></label><label>该列的原始单位<select data-node="${esc(r.node)}" data-role="unit"><option value="">必须明确选择</option>${r.units.map(u => `<option value="${esc(u)}">${esc(u === '1' ? '1（无量纲）' : u)}</option>`).join('')}</select></label></div>`).join('');
        document.getElementById('graph-lab-heldout-count').value = Math.min(20, fields.snapshot_rows);
        document.getElementById('graph-lab-heldout-source').textContent = `来源 ${fields.rows} 行；已固定第 ${fields.snapshot_start+1}–${fields.snapshot_start+fields.snapshot_rows} 行验证快照。当前表的后续更改不会影响此快照；需要其他范围时，修改起始行并重新读取字段。`;
        graphLabMessage('验证字段已读取，尚未计算。请核对列、原始单位与数据独立性，再执行只读审计。');
    } catch (error) { graphLabMessage(error.message); }
    finally { graphLabUI.busy = false; graphLabControls(); }
}
async function graphLabConfirmTable() {
    if (graphLabUI.busy || graphLabUI.state?.status !== 'ready') return;
    if (graphLabUI.heldoutContext?.id !== graphLabUI.id) return graphLabMessage('请先读取当前验证表字段。');
    graphLabUI.busy = true; graphLabControls();
    try {
        const bindings = Object.create(null);
        document.querySelectorAll('#graph-lab-heldout-mapping select').forEach(el => {
            if (!bindings[el.dataset.node]) bindings[el.dataset.node] = {};
            bindings[el.dataset.node][el.dataset.role] = el.value;
        });
        const config = {bindings, table_snapshot: graphLabUI.heldoutContext.token,
            start_row: Number(document.getElementById('graph-lab-heldout-start').value)-1,
            row_count: Number(document.getElementById('graph-lab-heldout-count').value),
            si_contract_confirmed: document.getElementById('graph-lab-heldout-si').checked};
        await graphLabRequest(`/api/graph-lab/${graphLabUI.id}/confirm-table`, config);
        graphLabUI.state.status = 'confirming';
        graphLabMessage('表格已映射到冻结变量，开始只读审计；不会重新拟合或搜索。');
        await graphLabRefresh();
    } catch (error) {
        const errors = {si_numeric_contract_confirmation_required:'请先确认冻结数值的 SI 约定与验证数据来源。',
            heldout_unit_dimension_mismatch:'单位不匹配或未选择；不能把时间当长度或将未知单位当无量纲。',
            numeric_measure_column_required:'请选择当前验证表中存在的数值列，不能使用编号。',
            distinct_heldout_columns_required:'每个输入/目标必须对应不同的列。', invalid_heldout_window:'验证范围必须是当前表内的 1–256 行。',
            stale_heldout_table_snapshot:'验证表快照已失效，请重新读取字段并核对映射。',
            holdout_development_overlap:'验证点与开发数据或探针重叠，不能用于独立确认。', duplicate_holdout_point:'验证范围包含重复输入点，当前不能作为独立样本处理。',
            point_outside_domain:'验证输入超出冻结模型的声明区间。本入口不自动扩大适用域。', invalid_heldout_table_values:'验证表有缺失、非有限或过大的数值，不会自动补齐或丢弃。'};
        graphLabMessage(errors[error.message] || error.message);
    } finally { graphLabUI.busy = false; graphLabControls(); }
}
