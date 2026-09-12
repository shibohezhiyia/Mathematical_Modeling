/* Presentation only: no repair requests or model calls are issued here. */
function renderModelDiagnostics(panel) {
    if (!panel || panel.schema_version !== 'mathmodel.model-diagnostics/v1') return '';
    const records = (panel.records || []).slice(0, 64);
    const categories = {resource: '资源', numerical: '数值', parameter: '参数', structure: '结构假设',
        observation: '观测假设', semantic: '题意/绑定', data: '数据', dependency: '上游依赖',
        execution: '执行待定位', validation: '最终验证'};
    const phases = {development: '训练/选参', execution: '执行', preflight: '编译前', final_test: '锁定测试 · 只读'};
    const searchCount = ((panel.search_feedback || {}).records || []).length;
    const finalCount = records.filter(record => (record.context || {}).phase === 'final_test').length;
    const coverage = panel.coverage || {};
    const residuals = (panel.development_checks || {}).residuals || [];
    const unclosed = (panel.development_checks || {}).unclosed_state_competition || {};
    const unclosedCandidates = (unclosed.candidate_explanations || []).slice(0, 8);
    const insufficient = residuals.filter(item => item.pattern_status === 'insufficient_disjoint_windows').length;
    let output = '<h4>诊断与修复方向 <span class="research-risk">仅建议 · 未自动改模型</span></h4>';
    output += `<div class="research-metrics"><span><small>开发/执行反馈</small><strong>${searchCount}</strong></span><span><small>最终审计 · 不回传搜索</small><strong>${finalCount}</strong></span><span><small>预测模型诊断</small><strong>${Number(coverage.prediction_models_with_diagnostics || 0)}</strong></span><span><small>聚类审计</small><strong>${Number(coverage.clustering_models_checked || 0)}</strong></span><span><small>时序结构集</small><strong>${Number(coverage.temporal_structure_sets_checked || 0)}</strong></span></div>`;
    output += '<p class="hint">阈值用于探索性筛查，不是显著性检验。误差大、残差相关或进程退出，都不能单独证明存在隐变量或模型结构错误。</p>';
    if (insufficient) output += `<p class="research-warning">${insufficient} 个状态的非重叠选参窗口不足，暂不判断其残差模式。</p>`;
    if (unclosed.status === 'screening_only' && unclosedCandidates.length) {
        output += '<h5>未闭合系统竞争筛查</h5><p class="hint">分数只是开发段筛查信号，不是概率、因果结论或隐状态证明；候选需要独立轨迹和观测模型继续验证。</p>';
        output += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>候选解释</th><th>状态</th><th>筛查分数</th><th>不代表</th></tr></thead><tbody>';
        unclosedCandidates.forEach(candidate => {
            output += `<tr><td>${escapeHtml(`${candidate.id || '-'} · ${candidate.state || '-'}`)}</td><td>${escapeHtml(candidate.status || 'candidate')}</td><td>${Number(candidate.score || 0).toFixed(3)}</td><td>${escapeHtml((candidate.not_claimed || []).join('、'))}</td></tr>`;
        });
        output += '</tbody></table></div>';
    }
    const structureChecks = (panel.structure_checks || []).slice(0, 32);
    if (structureChecks.length) {
        output += '<h5>时序结构候选信号</h5><p class="hint">周期、变点和单调性只是有序开发段筛查信号，不证明机制切换、因果关系或域外外推。</p>';
        output += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>序列</th><th>信号</th></tr></thead><tbody>';
        structureChecks.forEach(check => {
            output += `<tr><td>${escapeHtml(check.subject || '-')}</td><td>${escapeHtml(check.signal || '-')}</td></tr>`;
        });
        output += '</tbody></table></div>';
    }
    if (!records.length) {
        output += '<p class="hint">当前接入范围未发现可定位问题，不代表模型正确；尚未覆盖全部模型与验证方式。</p>';
        return output;
    }
    const actions = panel.proposed_actions || [];
    output += '<div class="table-wrapper"><table class="data-table"><thead><tr><th>范围 / 类型</th><th>发现及边界</th><th>下一步建议</th></tr></thead><tbody>';
    records.forEach(record => {
        const context = record.context || {};
        const related = actions.filter(action => action.diagnostic_id === record.id).slice(0, 4);
        const advice = related.map(action => `<p>${escapeHtml(action.label || '')}<br><small>${escapeHtml(action.guard || '')}</small></p>`).join('');
        const alternatives = (record.alternative_explanations || []).map(item => escapeHtml(item)).join('；');
        output += `<tr><td>${escapeHtml(phases[context.phase] || '未定位')}<br>${escapeHtml(categories[record.category] || '待核查')}<br><small>${escapeHtml(String(context.subject ?? ''))}</small></td>`;
        output += `<td>${escapeHtml(record.summary || '')}${alternatives ? `<p class="hint">其他解释：${alternatives}</p>` : ''}<details><summary>查看诊断依据</summary><pre class="code-block">${escapeHtml(JSON.stringify(record.evidence || {}, null, 2))}</pre></details></td><td>${advice || '保留未评估状态。'}</td></tr>`;
    });
    output += '</tbody></table></div><p class="hint">最终测试只用于判决。若据其结果改模型，须另设未使用的确认数据，不能在同一测试集反复试到通过。</p>';
    if (panel.dropped_diagnostics) output += '<p class="research-warning">诊断数量达到上限，当前仅展示部分记录；不能视为全项目检查完成。</p>';
    return output;
}
