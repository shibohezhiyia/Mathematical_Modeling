// Run with: node tests/test_model_diagnostics_ui.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
})[char]);
const context = {escapeHtml};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/static/js/model_diagnostics.js'), 'utf8'), context);
const panel = {
    schema_version: 'mathmodel.model-diagnostics/v1',
    records: [{id: 'x', category: 'structure', summary: '<img src=x onerror=alert(1)>',
        context: {phase: 'final_test', subject: '<script>unsafe</script>'},
        evidence: {detail: '<svg onload=alert(1)>'}, alternative_explanations: ['<iframe>']}],
    proposed_actions: [{diagnostic_id: 'x', label: '<button>执行</button>', guard: '<script>bad</script>'}],
    search_feedback: {records: []}
};
const output = context.renderModelDiagnostics(panel);
assert(output.includes('最终审计 · 不回传搜索'));
assert(output.includes('锁定测试 · 只读'));
assert(output.includes('仅建议 · 未自动改模型'));
assert(output.includes('&lt;img'));
assert(!/<(img|svg|script|iframe|button)/i.test(output));
const screening = context.renderModelDiagnostics({
    ...panel,
    development_checks: {unclosed_state_competition: {
        status: 'screening_only',
        candidate_explanations: [{id: 'memory_or_latent_state', state: 'x', score: 0.7,
            not_claimed: ['hidden_state_exists']}],
    }},
});
assert(screening.includes('未闭合系统竞争筛查'));
assert(screening.includes('hidden_state_exists'));
const empty = context.renderModelDiagnostics({schema_version: panel.schema_version, records: []});
assert(empty.includes('不代表模型正确'));
assert.equal(context.renderModelDiagnostics(null), '');
const template = fs.readFileSync(path.join(__dirname, '../web/templates/index.html'), 'utf8');
assert(template.indexOf('js/model_diagnostics.js') < template.indexOf('js/app.js'));
console.log('model diagnostics UI: escaped, read-only and correctly loaded');
