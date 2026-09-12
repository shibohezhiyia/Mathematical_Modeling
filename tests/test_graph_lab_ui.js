const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/static/js/graph_lab.js'), 'utf8'), context);
const state = {id:'a'.repeat(32), status:'ready', candidate_count:1, selected_hash:'<svg onload=alert(1)>',
    reports:[{hypothesis_hash:'<img src=x>', status:'eligible_for_confirmation', search_rmse:'<script>bad</script>'}],
    downloads:['report','../../secret','__proto__']};
const output = context.renderGraphLab(state);
assert(output.includes('候选已冻结') && output.includes('尚未验证'));
assert(output.includes('&lt;svg') && !/<(svg|script|img)/.test(output));
assert(!output.includes('../../secret') && !output.includes('artifact/__proto__'));
assert(output.includes('artifact/report'));
const final = context.renderGraphLab({...state, status:'done', confirmation:{status:'failed_heldout_checks', metrics:[]}});
assert(final.includes('留出检查未通过') && !final.includes('有限留出检查通过（不是数学证明）'));
const incomplete = context.renderGraphLab({...state, confirmation:{status:'execution_incomplete'}});
assert(incomplete.includes('不能判定模型错误'));
assert(context.graphLabFitLabel({status:'fit_budget_exhausted'}).includes('未判定结构错误'));
assert(context.graphLabFitLabel({status:'local_fit_completed', attempts:[{}, {}]}).includes('尝试 2 个起点'));
assert(context.graphLabFitLabel({status:'bounded_linear_fit_completed', attempts:[{}]}).includes('结构证明为仿射'));
assert(context.graphLabFitLabel({status:'__proto__'}) === '无拟合记录');
assert(context.graphLabFitLabel({status:'<script>'}) === '无拟合记录');
const template = fs.readFileSync(path.join(__dirname, '../web/templates/index.html'), 'utf8');
assert(template.includes('id="graph-lab-heldout" accept=".json,application/json" disabled'));
assert(template.includes('js/graph_lab.js'));
console.log('graph lab: staged status, escaped rendering and allowlisted downloads passed');
