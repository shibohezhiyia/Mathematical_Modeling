"""
智能图表 Web 前端

Flask 后端 API，为单页应用提供数据服务。

启动方式:
    cd web && python app.py
    # 或
    python -m web.app

访问: http://localhost:5000
"""

import os
import sys
import json
import uuid
import base64
import threading
import io
import logging
import traceback
import time
import re
from functools import wraps
from typing import Dict, Any, Optional
from dataclasses import asdict
from pathlib import Path

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, render_template, send_file, session
from werkzeug.exceptions import HTTPException
import pandas as pd
import numpy as np

from core.data_module import DataModule
from core.modeling_engine import ModelingEngine, TaskType, TaskTypeDetector, CrossValidator
from core.integrated_pipeline import IntegratedPipeline
from core.evaluation_engine import print_decision_report, DecisionMode
from core.visualization import (
    DataVisualizer, ModelVisualizer, EvaluationVisualizer,
    plot_data_profile, plot_modeling_summary
)
from core.performance_scheduler import PerformanceScheduler
from core.workspace_manager import get_workspace_manager, set_workspace_config

# 确保工作空间根目录始终指向项目根目录（避免 IDE/脚本启动时 cwd 不一致导致临时文件写到C盘）
set_workspace_config(root_dir=str(PROJECT_ROOT))

from extensions.llm_analyzer import LLMAnalyzer, LLMConfig, get_default_configs
from extensions.report_engine import (
    ReportEngine, ReportConfig, PivotConfig, CellConfig,
    AGG_FUNCTIONS, AGG_NAMES,
)
from extensions.advanced_analytics import AdvancedAnalytics
from utils.helpers import log_info, log_warning, log_error, get_log_store
from core.dependency_manager import (
    get_missing_dependencies,
    install_dependency,
    install_all_missing,
    OPTIONAL_DEPENDENCIES,
    check_package,
    DEFAULT_THIRD_PARTY_DIR,
)
from core.modeling_engine import ModelLibrary
from pathlib import Path

# -----------------------------------------------------------------------------
# 把 stdout/stderr 捕获到 LogStore，让前端日志面板实时显示服务器输出
# -----------------------------------------------------------------------------
class StreamInterceptor:
    """拦截 stdout/stderr，同时输出到原始流和 LogStore"""
    _inside = False
    def __init__(self, original):
        self.original = original
        self._buffer = ""
    def write(self, text):
        self.original.write(text)
        self.original.flush()
        if StreamInterceptor._inside or not text:
            return
        StreamInterceptor._inside = True
        try:
            self._buffer += text
            while '\n' in self._buffer:
                line, self._buffer = self._buffer.split('\n', 1)
                self._push(line)
        finally:
            StreamInterceptor._inside = False
    def _push(self, line):
        line = line.rstrip()
        if not line:
            return
        # 过滤 Werkzeug 访问日志，避免刷屏
        if re.match(r'^\d+\.\d+\.\d+\.\d+', line):
            return
        if ' - - [' in line and ('"GET ' in line or '"POST ' in line or '"PUT ' in line):
            return
        try:
            get_log_store().add("INFO", line, "系统")
        except Exception:
            pass
    def flush(self):
        self.original.flush()
        if self._buffer.strip():
            self._push(self._buffer)
            self._buffer = ""

sys.stdout = StreamInterceptor(sys.__stdout__)
sys.stderr = StreamInterceptor(sys.__stderr__)

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.secret_key = 'intelligent_charts_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# ------------------------------------------------------------------------------
# 日志配置：同时输出到控制台和文件
# ------------------------------------------------------------------------------
_LOG_DIR = PROJECT_ROOT / 'data' / 'logs'
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / 'flask_api.log'

_file_handler = logging.FileHandler(_log_file, encoding='utf-8')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
))

app.logger.setLevel(logging.INFO)
app.logger.addHandler(_file_handler)
app.logger.addHandler(_console_handler)

# 避免重复日志（Flask 默认会添加一个 StreamHandler）
app.logger.propagate = False

log_info(f"Flask API 日志已配置: {_log_file}", category="System")

# 全局会话存储（内存，生产环境应使用 Redis）
user_sessions: Dict[str, Dict[str, Any]] = {}

UPLOAD_DIR = PROJECT_ROOT / 'data' / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_session() -> Dict[str, Any]:
    """获取当前会话数据"""
    sid = session.get('sid')
    if not sid:
        sid = str(uuid.uuid4())
        session['sid'] = sid
        user_sessions[sid] = {}
    if sid not in user_sessions:
        user_sessions[sid] = {}
    sdata = user_sessions[sid]
    # 初始化训练事件存储（用于实时透明化）
    if 'train_events' not in sdata:
        sdata['train_events'] = []
    if 'train_live_results' not in sdata:
        sdata['train_live_results'] = []
    return sdata


def clear_session():
    """清空当前会话"""
    sid = session.get('sid')
    if sid and sid in user_sessions:
        # 清理上传文件
        sdata = user_sessions[sid]
        if 'upload_path' in sdata:
            try:
                os.remove(sdata['upload_path'])
            except:
                pass
        user_sessions[sid] = {}
        user_sessions[sid]['train_events'] = []
        user_sessions[sid]['train_live_results'] = []


def clean_for_json(obj: Any) -> Any:
    """递归清理对象中的 NaN/Inf/numpy 类型，使其可序列化为标准 JSON"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    # numpy 字符串类型（如 np.str_）
    if hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.str_):
        return str(obj)
    # pandas Timestamp / datetime
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj


def api_error_response(error: str, detail: str = None, status_code: int = 500):
    """统一记录 API 错误并返回标准错误响应"""
    endpoint = request.endpoint or 'unknown'
    msg = f"[{endpoint}] {error}"
    full_detail = detail or traceback.format_exc()
    app.logger.error(f"{msg}\n{full_detail}")
    log_error(msg, category="API")
    payload = {'success': False, 'error': error, 'endpoint': endpoint}
    if detail:
        payload['detail'] = detail
    return jsonify(payload), status_code


def df_to_dict(df: pd.DataFrame, max_rows: int = 20) -> Dict[str, Any]:
    """DataFrame 转为前端可用的字典"""
    preview_df = df.head(max_rows)
    return clean_for_json({
        'columns': df.columns.tolist(),
        'dtypes': {c: str(df[c].dtype) for c in df.columns},
        'shape': df.shape,
        'preview': preview_df.to_dict('records'),
        'memory_mb': round(df.memory_usage(deep=True).sum() / (1024**2), 2),
        'missing': {c: int(df[c].isnull().sum()) for c in df.columns},
    })


# =============================================================================
# 页面路由
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


# =============================================================================
# 数据上传 API
# =============================================================================

def _read_file_with_sheets(save_path, ext):
    """读取文件，对于Excel返回所有sheet名称"""
    sheets = None
    if ext in ('.xls', '.xlsx'):
        xl = pd.ExcelFile(save_path)
        sheets = xl.sheet_names
        df = pd.read_excel(save_path, sheet_name=sheets[0])
    elif ext == '.csv':
        df = pd.read_csv(save_path)
    elif ext == '.json':
        df = pd.read_json(save_path)
    elif ext == '.parquet':
        df = pd.read_parquet(save_path)
    else:
        raise ValueError(f'不支持的格式: {ext}')
    return df, sheets


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """上传数据文件（支持多文件和多sheet Excel）"""
    files = request.files.getlist('files') or []
    if not files:
        # 兼容旧版单文件上传
        single = request.files.get('file')
        if single:
            files = [single]
    
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({'success': False, 'error': '未找到文件'}), 400
    
    sdata = get_session()
    # 不清空session，追加数据集实现连续上传
    existing = sdata.get('uploaded_files', [])
    
    uploaded_files = list(existing)
    try:
        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()
            save_name = f"{uuid.uuid4().hex}{ext}"
            save_path = UPLOAD_DIR / save_name
            file.save(save_path)
            
            df, sheets = _read_file_with_sheets(save_path, ext)
            
            file_info = {
                'save_name': save_name,
                'filename': file.filename,
                'ext': ext,
                'path': str(save_path),
                'shape': clean_for_json(list(df.shape)),
                'sheets': sheets,
                'active_sheet': sheets[0] if sheets else None,
                'columns': list(df.columns),
            }
            uploaded_files.append(file_info)
        
        # 存储多文件信息
        sdata['uploaded_files'] = uploaded_files
        
        # 加载新上传的第一个文件作为当前数据集
        new_first = uploaded_files[len(existing)] if existing else uploaded_files[0]
        df_first, _ = _read_file_with_sheets(new_first['path'], new_first['ext'])
        sdata['df'] = df_first
        sdata['active_file_index'] = len(existing) if existing else 0
        sdata['active_sheet'] = new_first.get('active_sheet')
        sdata['df_info'] = df_to_dict(df_first)
        
        # 自动推断目标列
        target_hint = None
        for col in reversed(df_first.columns):
            missing_rate = df_first[col].isnull().sum() / len(df_first)
            if 0.05 < missing_rate < 0.95:
                target_hint = col
                break
        
        return jsonify(clean_for_json({
            'success': True,
            'data': df_to_dict(df_first),
            'files': uploaded_files,
            'target_hint': target_hint,
            'appended': len(existing) > 0,
        }))
    except Exception as e:
        return api_error_response(str(e))


def _clear_analysis_results(sdata):
    """切换数据集时清除旧的分析结果（需要重新分析）"""
    keys_to_clear = [
        'eda_data', 'model_result', 'train_error', 'train_error_stack',
        'train_config', 'best_model', 'llm_analysis_result',
        'llm_analysis_status', 'llm_analysis_error',
    ]
    for k in keys_to_clear:
        sdata.pop(k, None)


@app.route('/api/data/quality', methods=['POST'])
def api_data_quality():
    """Generate data quality report"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': 'No data uploaded'}), 400
    data = request.get_json() or {}
    target_col = data.get('target_col')
    task_type = data.get('task_type')
    try:
        from core.data_quality import generate_data_quality_report
        report = generate_data_quality_report(df, target_col=target_col, task_type=task_type)
        return jsonify({'success': True, 'report': report})
    except Exception as e:
        log_error(f'[DataQuality] {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/problem/analyze', methods=['POST'])
def api_problem_analyze():
    """Analyze a math modeling problem description"""
    data = request.get_json() or {}
    description = data.get('description', '')
    if not description:
        return jsonify({'success': False, 'error': 'Description required'}), 400
    try:
        from core.problem_solver import analyze_problem
        result = analyze_problem(description)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/generate', methods=['POST'])
def api_data_generate():
    """Generate synthetic dataset for modeling without provided data"""
    data = request.get_json() or {}
    try:
        from core.synthetic_data_generator import generate_synthetic_data, generate_from_description
        if data.get('description'):
            df = generate_from_description(data['description'], **{k: v for k, v in data.items() if k != 'description'})
        else:
            df = generate_synthetic_data(
                task_type=data.get('task_type', 'classification'),
                n_samples=data.get('n_samples', 1000),
                n_features=data.get('n_features', 10),
                n_classes=data.get('n_classes'),
                noise=data.get('noise', 0.1),
                random_state=data.get('random_state', 42)
            )
        sdata = get_session()
        sdata['df'] = df
        sdata['uploaded_files'] = [{'filename': 'synthetic_data.csv', 'shape': list(df.shape), 'columns': list(df.columns)}]
        # Auto-detect target column
        for col in ['target', 'cluster']:
            if col in df.columns:
                sdata['target_col'] = col
                break
        return jsonify({'success': True, 'shape': list(df.shape), 'columns': list(df.columns), 'target_col': sdata.get('target_col')})
    except Exception as e:
        log_error(f'[SyntheticData] {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/autofix', methods=['POST'])
def api_data_autofix():
    """Auto-fix data quality issues"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': 'No data uploaded'}), 400
    data = request.get_json() or {}
    try:
        from core.data_quality import generate_data_quality_report
        from core.data_autofix import autofix_dataframe
        target_col = data.get('target_col')
        task_type = data.get('task_type')
        report = generate_data_quality_report(df, target_col=target_col, task_type=task_type)
        fixed_df, fixes = autofix_dataframe(
            df, report=report, target_col=target_col,
            drop_high_missing=data.get('drop_high_missing', True),
            missing_threshold=data.get('missing_threshold', 50.0),
            fix_outliers=data.get('fix_outliers', False),
            drop_duplicates=data.get('drop_duplicates', True)
        )
        sdata['df'] = fixed_df
        sdata['autofix_applied'] = True
        sdata['autofix_log'] = fixes
        return jsonify({'success': True, 'fixes': fixes, 'n_rows': len(fixed_df), 'n_columns': len(fixed_df.columns)})
    except Exception as e:
        log_error(f'[DataAutofix] {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/upload/select', methods=['POST'])
def api_upload_select():
    """选择指定文件的指定sheet进行分析"""
    sdata = get_session()
    data = request.get_json() or {}
    file_index = data.get('file_index', 0)
    sheet_name = data.get('sheet_name')
    
    files = sdata.get('uploaded_files', [])
    if not files or file_index >= len(files):
        return jsonify({'success': False, 'error': '文件不存在'}), 400
    
    file_info = files[file_index]
    try:
        if file_info['ext'] in ('.xls', '.xlsx'):
            df = pd.read_excel(file_info['path'], sheet_name=sheet_name)
        else:
            df, _ = _read_file_with_sheets(file_info['path'], file_info['ext'])
        
        # 切换数据集时清除旧分析结果
        _clear_analysis_results(sdata)
        
        sdata['df'] = df
        sdata['active_file_index'] = file_index
        sdata['active_sheet'] = sheet_name
        sdata['df_info'] = df_to_dict(df)
        
        # 更新激活状态
        for i, f in enumerate(files):
            f['active'] = (i == file_index)
            if i == file_index:
                f['active_sheet'] = sheet_name
        
        target_hint = None
        for col in reversed(df.columns):
            missing_rate = df[col].isnull().sum() / len(df)
            if 0.05 < missing_rate < 0.95:
                target_hint = col
                break
        
        return jsonify({
            'success': True,
            'data': df_to_dict(df),
            'file_index': file_index,
            'sheet_name': sheet_name,
            'target_hint': target_hint,
        })
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/datasets', methods=['GET'])
def api_datasets():
    """获取所有已上传的数据集列表"""
    sdata = get_session()
    files = sdata.get('uploaded_files', [])
    active_idx = sdata.get('active_file_index', 0)
    
    datasets = []
    for i, f in enumerate(files):
        datasets.append({
            'index': i,
            'filename': f['filename'],
            'shape': f['shape'],
            'sheets': f.get('sheets'),
            'active_sheet': f.get('active_sheet'),
            'columns': f.get('columns', []),
            'is_active': i == active_idx,
        })
    
    return jsonify({'success': True, 'datasets': datasets, 'active_index': active_idx})


@app.route('/api/datasets/<int:index>', methods=['DELETE'])
def api_delete_dataset(index):
    """删除指定数据集"""
    sdata = get_session()
    files = sdata.get('uploaded_files', [])
    
    if index < 0 or index >= len(files):
        return jsonify({'success': False, 'error': '数据集不存在'}), 400
    
    removed = files.pop(index)
    # 尝试删除物理文件
    try:
        if os.path.exists(removed['path']):
            os.remove(removed['path'])
    except:
        pass
    
    # 如果删除的是当前活跃数据集，切换到第一个
    active_idx = sdata.get('active_file_index', 0)
    if active_idx == index and files:
        sdata['active_file_index'] = 0
        first = files[0]
        df, _ = _read_file_with_sheets(first['path'], first['ext'])
        sdata['df'] = df
        sdata['df_info'] = df_to_dict(df)
        sdata['active_sheet'] = first.get('active_sheet')
        _clear_analysis_results(sdata)
    elif not files:
        # 全部删除了，清空相关数据
        sdata['df'] = None
        sdata['df_info'] = None
        sdata['active_file_index'] = None
        sdata['active_sheet'] = None
        _clear_analysis_results(sdata)
    elif active_idx > index:
        sdata['active_file_index'] = active_idx - 1
    
    sdata['uploaded_files'] = files
    return jsonify({'success': True, 'datasets_count': len(files)})


@app.route('/api/upload/merge', methods=['POST'])
def api_upload_merge():
    """合并多个sheet/文件的数据"""
    sdata = get_session()
    data = request.get_json() or {}
    sources = data.get('sources', [])  # [{file_index, sheet_name}]
    axis = data.get('axis', 0)  # 0=纵向(行), 1=横向(列)
    
    if len(sources) < 2:
        return jsonify({'success': False, 'error': '至少需要选择2个表进行合并'}), 400
    
    files = sdata.get('uploaded_files', [])
    dfs = []
    try:
        for src in sources:
            idx = src.get('file_index', 0)
            sheet = src.get('sheet_name')
            if idx >= len(files):
                continue
            fi = files[idx]
            if fi['ext'] in ('.xls', '.xlsx') and sheet:
                df = pd.read_excel(fi['path'], sheet_name=sheet)
            else:
                df, _ = _read_file_with_sheets(fi['path'], fi['ext'])
            dfs.append(df)
        
        if axis == 0:
            merged = pd.concat(dfs, axis=0, ignore_index=True)
        else:
            merged = pd.concat(dfs, axis=1)
        
        sdata['df'] = merged
        sdata['df_info'] = df_to_dict(merged)
        sdata['merged_from'] = sources
        
        target_hint = None
        for col in reversed(merged.columns):
            missing_rate = merged[col].isnull().sum() / len(merged)
            if 0.05 < missing_rate < 0.95:
                target_hint = col
                break
        
        return jsonify({
            'success': True,
            'data': df_to_dict(merged),
            'shape': [merged.shape[0], merged.shape[1]],
            'target_hint': target_hint,
        })
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/upload/join', methods=['POST'])
def api_upload_join():
    """多表关联（SQL风格 join）"""
    sdata = get_session()
    data = request.get_json() or {}
    left = data.get('left', {})
    right = data.get('right', {})
    on = data.get('on', '')
    how = data.get('how', 'inner')  # inner, left, right, outer
    
    if not on:
        return jsonify({'success': False, 'error': '请指定关联键'}), 400
    
    files = sdata.get('uploaded_files', [])
    try:
        li = left.get('file_index', 0)
        ls = left.get('sheet_name')
        ri = right.get('file_index', 0)
        rs = right.get('sheet_name')
        
        # 读取左表
        lf = files[li]
        if lf['ext'] in ('.xls', '.xlsx') and ls:
            df_left = pd.read_excel(lf['path'], sheet_name=ls)
        else:
            df_left, _ = _read_file_with_sheets(lf['path'], lf['ext'])
        
        # 读取右表
        rf = files[ri]
        if rf['ext'] in ('.xls', '.xlsx') and rs:
            df_right = pd.read_excel(rf['path'], sheet_name=rs)
        else:
            df_right, _ = _read_file_with_sheets(rf['path'], rf['ext'])
        
        joined = df_left.merge(df_right, on=on, how=how, suffixes=('', '_right'))
        
        sdata['df'] = joined
        sdata['df_info'] = df_to_dict(joined)
        
        target_hint = None
        for col in reversed(joined.columns):
            missing_rate = joined[col].isnull().sum() / len(joined)
            if 0.05 < missing_rate < 0.95:
                target_hint = col
                break
        
        return jsonify({
            'success': True,
            'data': df_to_dict(joined),
            'shape': [joined.shape[0], joined.shape[1]],
            'target_hint': target_hint,
        })
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/data/info', methods=['GET'])
def api_data_info():
    """获取数据基本信息"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    
    # 更详细的类型分析
    type_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique(dropna=True)
        missing = int(df[col].isnull().sum())
        missing_rate = missing / len(df)
        
        inferred = 'numeric'
        if dtype == 'object':
            inferred = 'text' if (n_unique > 50 or df[col].astype(str).str.len().mean() > 30) else 'categorical'
        elif 'datetime' in dtype:
            inferred = 'datetime'
        elif 'bool' in dtype:
            inferred = 'boolean'
        
        type_info.append({
            'column': col,
            'dtype': dtype,
            'inferred_type': inferred,
            'n_unique': int(n_unique),
            'missing': missing,
            'missing_rate': round(missing_rate, 4),
        })
    
    return jsonify({
        'success': True,
        'info': {
            'shape': df.shape,
            'memory_mb': round(df.memory_usage(deep=True).sum() / (1024**2), 2),
            'columns': type_info,
        }
    })


@app.route('/api/data/eda', methods=['GET'])
def api_data_eda():
    """EDA 分析：数值统计、相关性等"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 数值统计
    stats = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        stats = {col: {
            'mean': round(desc.loc[col, 'mean'], 4) if col in desc.index else None,
            'std': round(desc.loc[col, 'std'], 4) if col in desc.index else None,
            'min': round(desc.loc[col, 'min'], 4) if col in desc.index else None,
            'max': round(desc.loc[col, 'max'], 4) if col in desc.index else None,
            'median': round(numeric_df[col].median(), 4),
        } for col in numeric_df.columns}
    
    # 相关性矩阵（前20个数值列）
    corr = {}
    if numeric_df.shape[1] >= 2:
        corr_df = numeric_df.iloc[:, :20].corr()
        corr = {
            'columns': corr_df.columns.tolist(),
            'values': corr_df.values.tolist()
        }
    
    # 类别计数（前5个类别列）
    cat_counts = {}
    cat_cols = df.select_dtypes(include=['object', 'category']).columns[:5]
    for col in cat_cols:
        cat_counts[col] = df[col].value_counts().head(10).to_dict()
    
    eda_payload = {
        'success': True,
        'eda': {
            'statistics': stats,
            'correlation': corr,
            'categorical_counts': cat_counts,
        }
    }
    sdata['eda_data'] = eda_payload['eda']
    return jsonify(clean_for_json(eda_payload))


# =============================================================================
# 数据筛选与自动分析 API
# =============================================================================

@app.route('/api/data/column-quality', methods=['GET'])
def api_column_quality():
    """获取各列质量分析，用于列筛选决策"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    
    try:
        quality = []
        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            n_total = len(s)
            n_missing = int(s.isnull().sum())
            missing_rate = n_missing / n_total if n_total > 0 else 0
            n_unique = s.nunique(dropna=True)
            unique_rate = n_unique / (n_total - n_missing) if (n_total - n_missing) > 0 else 0
            
            is_numeric = pd.api.types.is_numeric_dtype(s)
            variance = float(s.var()) if is_numeric and n_total > 1 else None
            
            # 自动推荐
            reasons = []
            recommendation = 'keep'
            if missing_rate > 0.9:
                recommendation = 'drop'
                reasons.append('缺失率过高(>90%)')
            elif unique_rate == 1.0 and n_unique == (n_total - n_missing):
                recommendation = 'drop'
                reasons.append('唯一值过多(可能是ID列)')
            elif is_numeric and variance is not None and variance == 0:
                recommendation = 'drop'
                reasons.append('方差为0(无变化)')
            elif missing_rate > 0.5:
                recommendation = 'review'
                reasons.append('缺失率较高(>50%)')
            elif unique_rate == 1.0 and n_unique <= 2:
                recommendation = 'review'
                reasons.append('常数列或仅2个值')
            
            quality.append({
                'column': col,
                'dtype': dtype,
                'is_numeric': is_numeric,
                'count': n_total,
                'missing': n_missing,
                'missing_rate': round(missing_rate, 4),
                'unique': int(n_unique),
                'unique_rate': round(unique_rate, 4),
                'variance': round(variance, 6) if variance is not None else None,
                'recommendation': recommendation,
                'reasons': reasons,
            })
        
        return jsonify({'success': True, 'quality': quality})
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/data/filter', methods=['POST'])
def api_data_filter():
    """应用行列筛选"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    
    data = request.get_json() or {}
    columns = data.get('columns')  # 保留的列名列表
    row_filters = data.get('row_filters', [])  # 行筛选条件 [{column, operator, value}]
    
    try:
        # 备份原始数据（第一次筛选时备份）
        if 'df_original' not in sdata:
            sdata['df_original'] = df.copy()
        
        # 从原始数据开始（避免多次筛选叠加的混乱）
        df_work = sdata['df_original'].copy()
        
        # 1. 列筛选
        if columns and len(columns) > 0:
            valid_cols = [c for c in columns if c in df_work.columns]
            if len(valid_cols) == 0:
                return jsonify({'success': False, 'error': '未选择有效列'}), 400
            df_work = df_work[valid_cols]
        
        # 2. 行筛选
        for f in row_filters:
            col = f.get('column')
            op = f.get('operator')
            val = f.get('value')
            if not col or not op or col not in df_work.columns:
                continue
            
            col_series = df_work[col]
            is_num = pd.api.types.is_numeric_dtype(col_series)
            
            try:
                if is_num:
                    val = float(val)
            except:
                pass
            
            if op == 'eq':
                mask = col_series == val
            elif op == 'ne':
                mask = col_series != val
            elif op == 'gt':
                mask = col_series > val
            elif op == 'gte':
                mask = col_series >= val
            elif op == 'lt':
                mask = col_series < val
            elif op == 'lte':
                mask = col_series <= val
            elif op == 'contains':
                mask = col_series.astype(str).str.contains(str(val), na=False)
            elif op == 'startswith':
                mask = col_series.astype(str).str.startswith(str(val), na=False)
            elif op == 'endswith':
                mask = col_series.astype(str).str.endswith(str(val), na=False)
            elif op == 'isnull':
                mask = col_series.isnull()
            elif op == 'notnull':
                mask = col_series.notnull()
            else:
                continue
            
            df_work = df_work[mask]
        
        # 3. 更新session
        sdata['df'] = df_work
        sdata['df_info'] = df_to_dict(df_work)
        sdata['df_filtered'] = True
        sdata['active_columns'] = columns
        sdata['active_row_filters'] = row_filters
        
        # 清除旧的分析结果（数据变了需要重新分析）
        for k in ['eda_data', 'model_result', 'train_error', 'train_error_stack', 
                  'train_config', 'best_model', 'llm_analysis_result',
                  'llm_analysis_status', 'llm_analysis_error']:
            sdata.pop(k, None)
        
        # 自动推断目标列
        target_hint = None
        for col in reversed(df_work.columns):
            missing_rate = df_work[col].isnull().sum() / len(df_work)
            if 0.05 < missing_rate < 0.95:
                target_hint = col
                break
        
        return jsonify({
            'success': True,
            'data': df_to_dict(df_work),
            'shape': [df_work.shape[0], df_work.shape[1]],
            'target_hint': target_hint,
            'filtered': True,
            'row_count_before': sdata['df_original'].shape[0],
            'row_count_after': df_work.shape[0],
        })
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/data/reset-filter', methods=['POST'])
def api_reset_filter():
    """重置筛选，恢复原始数据"""
    sdata = get_session()
    if 'df_original' not in sdata:
        return jsonify({'success': False, 'error': '没有原始数据可恢复'}), 400
    
    df = sdata['df_original'].copy()
    sdata['df'] = df
    sdata['df_info'] = df_to_dict(df)
    sdata.pop('df_original', None)
    sdata.pop('df_filtered', None)
    sdata.pop('active_columns', None)
    sdata.pop('active_row_filters', None)
    
    # 清除旧的分析结果
    for k in ['eda_data', 'model_result', 'train_error', 'train_error_stack',
              'train_config', 'best_model', 'llm_analysis_result',
              'llm_analysis_status', 'llm_analysis_error']:
        sdata.pop(k, None)
    
    target_hint = None
    for col in reversed(df.columns):
        missing_rate = df[col].isnull().sum() / len(df)
        if 0.05 < missing_rate < 0.95:
            target_hint = col
            break
    
    return jsonify({
        'success': True,
        'data': df_to_dict(df),
        'shape': [df.shape[0], df.shape[1]],
        'target_hint': target_hint,
    })


# =============================================================================
# 高级统计分析 API
# =============================================================================

def _get_df_for_analytics():
    """辅助函数：获取当前DataFrame并校验"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return None, jsonify({'success': False, 'error': '请先上传数据'}), 400
    return df, None, None


@app.route('/api/analytics/summary', methods=['GET'])
def api_analytics_summary():
    """综合数据摘要"""
    df, err_resp, code = _get_df_for_analytics()
    if df is None:
        return err_resp, code
    try:
        analyzer = AdvancedAnalytics(df)
        result = analyzer.summary()
        return jsonify({'success': True, 'summary': clean_for_json(result)})
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/analytics/descriptive', methods=['POST'])
def api_analytics_descriptive():
    """扩展描述统计"""
    df, err_resp, code = _get_df_for_analytics()
    if df is None:
        return err_resp, code
    data = request.get_json() or {}
    columns = data.get('columns')
    try:
        analyzer = AdvancedAnalytics(df)
        results = analyzer.descriptive_stats(columns)
        return jsonify({
            'success': True,
            'stats': clean_for_json([asdict(r) for r in results]),
        })
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/analytics/pca', methods=['POST'])
def api_analytics_pca():
    """主成分分析"""
    df, err_resp, code = _get_df_for_analytics()
    if df is None:
        return err_resp, code
    data = request.get_json() or {}
    n_components = data.get('n_components')
    try:
        analyzer = AdvancedAnalytics(df)
        result = analyzer.pca_analysis(n_components)
        return jsonify({
            'success': True,
            'pca': clean_for_json(asdict(result)),
        })
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/analytics/factor', methods=['POST'])
def api_analytics_factor():
    """因子分析"""
    df, err_resp, code = _get_df_for_analytics()
    if df is None:
        return err_resp, code
    data = request.get_json() or {}
    n_factors = data.get('n_factors')
    rotation = data.get('rotation', 'varimax')
    try:
        analyzer = AdvancedAnalytics(df)
        result = analyzer.factor_analysis(n_factors, rotation)
        return jsonify({
            'success': True,
            'factor': clean_for_json(asdict(result)),
        })
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/analytics/correlation', methods=['POST'])
def api_analytics_correlation():
    """相关性矩阵分析"""
    df, err_resp, code = _get_df_for_analytics()
    if df is None:
        return err_resp, code
    data = request.get_json() or {}
    method = data.get('method', 'pearson')
    threshold = data.get('threshold', 0.5)
    try:
        analyzer = AdvancedAnalytics(df)
        result = analyzer.correlation_analysis(method, threshold)
        return jsonify({
            'success': True,
            'correlation': clean_for_json(asdict(result)),
        })
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/analytics/anova', methods=['POST'])
def api_analytics_anova():
    """方差分析"""
    df, err_resp, code = _get_df_for_analytics()
    if df is None:
        return err_resp, code
    data = request.get_json() or {}
    factor = data.get('factor')
    target = data.get('target')
    if not factor or not target:
        return jsonify({'success': False, 'error': '请指定 factor 和 target'}), 400
    try:
        analyzer = AdvancedAnalytics(df)
        result = analyzer.anova_analysis(factor, target)
        return jsonify({
            'success': True,
            'anova': clean_for_json(asdict(result)),
        })
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/analytics/chi2', methods=['POST'])
def api_analytics_chi2():
    """卡方检验"""
    df, err_resp, code = _get_df_for_analytics()
    if df is None:
        return err_resp, code
    data = request.get_json() or {}
    col1 = data.get('col1')
    col2 = data.get('col2')
    if not col1 or not col2:
        return jsonify({'success': False, 'error': '请指定 col1 和 col2'}), 400
    try:
        analyzer = AdvancedAnalytics(df)
        result = analyzer.chi2_analysis(col1, col2)
        return jsonify({'success': True, 'chi2': clean_for_json(result)})
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/analytics/outliers', methods=['POST'])
def api_analytics_outliers():
    """异常值检测"""
    df, err_resp, code = _get_df_for_analytics()
    if df is None:
        return err_resp, code
    data = request.get_json() or {}
    column = data.get('column')
    method = data.get('method', 'iqr')
    if not column:
        return jsonify({'success': False, 'error': '请指定 column'}), 400
    try:
        analyzer = AdvancedAnalytics(df)
        result = analyzer.outlier_detection(column, method)
        return jsonify({
            'success': True,
            'outliers': clean_for_json(asdict(result)),
        })
    except Exception as e:
        return api_error_response(str(e))


# =============================================================================
# 建模配置与训练 API
# =============================================================================

def _extract_model_result(result, mresult):
    """从 PipelineResult 和 ModelingResult 中提取模型结果摘要"""
    return {
        'task_type': mresult.task_type.value if hasattr(mresult.task_type, 'value') else str(mresult.task_type),
        'leaderboard': mresult.leaderboard.to_dict('records') if mresult.leaderboard is not None and not mresult.leaderboard.empty else [],
        'decision': {
            'mode': mresult.decision_report.mode.value if mresult.decision_report and hasattr(mresult.decision_report.mode, 'value') else '',
            'mode_description': mresult.decision_report.mode_description if mresult.decision_report else '',
            'recommended_model': mresult.decision_report.recommended_model if mresult.decision_report else '',
            'recommended_name': mresult.decision_report.recommended_name if mresult.decision_report else '',
            'recommendation_reason': mresult.decision_report.recommendation_reason if mresult.decision_report else '',
            'confidence': mresult.decision_report.confidence if mresult.decision_report else 0.0,
            'risks': mresult.decision_report.risks if mresult.decision_report else [],
            'scenario_advice': mresult.decision_report.scenario_advice if mresult.decision_report else '',
        } if mresult.decision_report else {},
        'preprocessing': {k: v for k, v in mresult.preprocessing_info.items() if k not in ('encoder', 'scaler')},
        'ensemble_weights': mresult.ensemble_result.get('weights') if mresult.ensemble_result else None,
        'optimization_history': mresult.optimization_history,
        'optimized_params': mresult.optimized_params,
        'automl_decision': {
            'meta_features': result.automl_recommendation.reasoning if result.automl_recommendation else '',
            'recommended_optimizer': result.automl_recommendation.optimizer if result.automl_recommendation else '',
            'recommended_models': result.automl_recommendation.model_keys if result.automl_recommendation else [],
            'recommended_ensemble': result.automl_recommendation.ensemble if result.automl_recommendation else '',
            'expected_time': result.automl_recommendation.expected_time if result.automl_recommendation else '',
        } if hasattr(result, 'automl_recommendation') and result.automl_recommendation else None,
    }

@app.route('/api/model/train', methods=['POST'])
def api_model_train():
    """启动模型训练"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    
    data = request.get_json() or {}
    target_col_raw = data.get('target_col')
    # 统一为多目标列表（向后兼容单字符串）
    if isinstance(target_col_raw, list):
        target_cols = [c for c in target_col_raw if c]
    elif isinstance(target_col_raw, str) and target_col_raw:
        target_cols = [target_col_raw]
    else:
        target_cols = []
    task_type = data.get('task_type')  # None=自动
    
    for col in target_cols:
        if col not in df.columns:
            return jsonify({'success': False, 'error': f'目标列 {col} 不存在'}), 400
    
    # 收集用户配置（手动覆盖）
    config = {
        'target_col': target_cols[0] if len(target_cols) == 1 else target_cols,
        'task_type': task_type,
        'model_keys': data.get('model_keys'),  # None=全部
        'encoding': data.get('encoding', 'auto'),
        'feature_selection': data.get('feature_selection', 'mi'),
        'ensemble': data.get('ensemble', 'weighted'),
        'feature_engineering': data.get('feature_engineering', False),
        'fold_type': data.get('fold_type', 'default'),
        'group_col': data.get('group_col'),
        'pseudo_labeling': data.get('pseudo_labeling', False),
        'pseudo_label_threshold': data.get('pseudo_label_threshold', 0.9),
        'n_splits': int(data.get('n_splits', 5)),
        'auto_decision_mode': data.get('auto_decision_mode', 'balanced'),
        'user_override_model': data.get('user_override_model'),
        'auto_sample': data.get('auto_sample', True),
        'max_samples': int(data.get('max_samples', 50000)),
        'optimize_hyperparams': data.get('optimize_hyperparams', False),
        'hyperparam_trials': int(data.get('hyperparam_trials', 20)),
        'visualization': False,  # Web 端单独请求图表
        'deep_learning': data.get('deep_learning'),
        'optimizer': data.get('optimizer', 'bayesian'),
        'dim_reduction': data.get('dim_reduction', 'none'),
    }
    
    # 在主线程获取 session ID（后台线程无法访问 Flask session）
    sid = session.get('sid')
    
    # 异步训练
    def train_task():
        train_start = time.time()
        try:
            set_workspace_config(root_dir=str(PROJECT_ROOT), allow_disk_write=True)
            
            # 进度回调：通过 session ID 更新全局会话存储
            def make_progress_callback(target_name):
                def progress_callback(step, current, total, message):
                    if sid and sid in user_sessions:
                        prefix = f"[{target_name}] " if len(target_cols) > 1 else ""
                        percent = round(current / total * 100, 1) if total > 0 else 0
                        user_sessions[sid]['train_progress'] = {
                            'step': step,
                            'current': current,
                            'total': total,
                            'message': prefix + message,
                            'percent': percent,
                            'target': target_name,
                        }
                        event = {
                            'id': len(user_sessions[sid].get('train_events', [])),
                            'timestamp': time.time(),
                            'step': step,
                            'current': current,
                            'total': total,
                            'message': prefix + message,
                            'percent': percent,
                            'target': target_name,
                        }
                        if 'train_events' not in user_sessions[sid]:
                            user_sessions[sid]['train_events'] = []
                        user_sessions[sid]['train_events'].append(event)
                        if step == 'model_done' and sid in user_sessions:
                            live_entry = {
                                'model_name': message.split(' CV done:')[0] if ' CV done:' in message else message,
                                'message': prefix + message,
                                'target': target_name,
                            }
                            if 'train_live_results' not in user_sessions[sid]:
                                user_sessions[sid]['train_live_results'] = []
                            user_sessions[sid]['train_live_results'].append(live_entry)
                return progress_callback
            
            # 单目标：保持原有行为；多目标：循环训练
            if len(target_cols) <= 1:
                # ===== 单目标 / 聚类模式 =====
                target_col = target_cols[0] if target_cols else None
                performance_config = config.get('performance', {}) if isinstance(config.get('performance'), dict) else {}
                pipeline = IntegratedPipeline(
                    strategy_preference=data.get('strategy_preference'),
                    target_col=target_col,
                    task_type=task_type,
                    model_keys=config['model_keys'],
                    encoding=config['encoding'],
                    feature_selection=config['feature_selection'],
                    ensemble=config['ensemble'],
                    feature_engineering=config.get('feature_engineering', False),
                    fold_type=config.get('fold_type', 'default'),
                    group_col=config.get('group_col'),
                    pseudo_labeling=config.get('pseudo_labeling', False),
                    pseudo_label_threshold=config.get('pseudo_label_threshold', 0.9),
                    n_splits=config['n_splits'],
                    optimize_hyperparams=config['optimize_hyperparams'],
                    hyperparam_trials=config['hyperparam_trials'],
                    auto_decision_mode=config['auto_decision_mode'],
                    user_override_model=config['user_override_model'],
                    auto_sample=config['auto_sample'],
                    max_samples=config['max_samples'],
                    deep_learning=config.get('deep_learning'),
                    optimizer=config.get('optimizer', 'bayesian'),
                    dim_reduction=config.get('dim_reduction', 'none'),
                    enable_kernel_approximation=performance_config.get('enable_kernel_approximation', config.get('enable_kernel_approximation', True)),
                    enable_precomputed_kernel_cache=performance_config.get('enable_precomputed_kernel_cache', config.get('enable_precomputed_kernel_cache', True)),
                    allow_disk_write=True,
                    progress_callback=make_progress_callback(target_col or 'clustering'),
                )
                result = pipeline.run(df)
                
                sdata['pipeline_result'] = result
                sdata['modeling_result'] = result.modeling_result
                sdata['multi_target_results'] = None
                sdata['train_status'] = 'done'
                sdata['train_error'] = None
                sdata['train_error_stack'] = None
                if result.target_col:
                    config['target_col'] = result.target_col
                sdata['train_config'] = config
                mresult = result.modeling_result
                sdata['model_result'] = _extract_model_result(result, mresult)
                log_info(f"[Web] 训练完成: {result.task_type}, 耗时 {result.total_time:.1f}s")
                
            else:
                # ===== 多目标模式 =====
                sdata['multi_target_results'] = {}
                all_leaderboards = []
                total_time = 0.0
                
                for idx, col in enumerate(target_cols):
                    log_info(f"[Web] 多目标训练 ({idx+1}/{len(target_cols)}): {col}")
                    performance_config = config.get('performance', {}) if isinstance(config.get('performance'), dict) else {}
                    pipeline = IntegratedPipeline(
                        strategy_preference=data.get('strategy_preference'),
                        target_col=col,
                        task_type=task_type,
                        model_keys=config['model_keys'],
                        encoding=config['encoding'],
                        feature_selection=config['feature_selection'],
                        ensemble=config['ensemble'],
                        feature_engineering=config.get('feature_engineering', False),
                        fold_type=config.get('fold_type', 'default'),
                        group_col=config.get('group_col'),
                        pseudo_labeling=config.get('pseudo_labeling', False),
                        pseudo_label_threshold=config.get('pseudo_label_threshold', 0.9),
                        n_splits=config['n_splits'],
                        optimize_hyperparams=config['optimize_hyperparams'],
                        hyperparam_trials=config['hyperparam_trials'],
                        auto_decision_mode=config['auto_decision_mode'],
                        user_override_model=config['user_override_model'],
                        auto_sample=config['auto_sample'],
                        max_samples=config['max_samples'],
                        deep_learning=config.get('deep_learning'),
                        optimizer=config.get('optimizer', 'bayesian'),
                        dim_reduction=config.get('dim_reduction', 'none'),
                        enable_kernel_approximation=performance_config.get('enable_kernel_approximation', config.get('enable_kernel_approximation', True)),
                        enable_precomputed_kernel_cache=performance_config.get('enable_precomputed_kernel_cache', config.get('enable_precomputed_kernel_cache', True)),
                        allow_disk_write=True,
                        progress_callback=make_progress_callback(col),
                    )
                    result = pipeline.run(df)
                    total_time += result.total_time
                    
                    # 提取该目标的结果摘要
                    mresult = result.modeling_result
                    summary = _extract_model_result(result, mresult)
                    summary['target_col'] = col
                    summary['total_time'] = result.total_time
                    sdata['multi_target_results'][col] = {
                        'pipeline_result': result,
                        'modeling_result': mresult,
                        'summary': summary,
                    }
                    all_leaderboards.append({
                        'target': col,
                        'leaderboard': summary.get('leaderboard', []),
                        'best_model': summary.get('decision', {}).get('recommended_name', ''),
                    })
                
                # 汇总多目标结果
                sdata['pipeline_result'] = list(sdata['multi_target_results'].values())[0]['pipeline_result']
                sdata['modeling_result'] = list(sdata['multi_target_results'].values())[0]['modeling_result']
                sdata['model_result'] = {
                    'multi_target': True,
                    'targets': target_cols,
                    'leaderboards': all_leaderboards,
                    'total_time': total_time,
                }
                sdata['train_status'] = 'done'
                sdata['train_error'] = None
                sdata['train_error_stack'] = None
                sdata['train_config'] = config
                log_info(f"[Web] 多目标训练完成: {len(target_cols)} 个目标, 总耗时 {total_time:.1f}s")
            
            # 记录实验到追踪器
            try:
                from core.experiment_tracker import log_experiment
                dataset_name = sdata.get('filename', '')
                duration = time.time() - train_start
                model_result = sdata.get('model_result', {})
                log_experiment(config=config, result=model_result, duration=duration, dataset_name=dataset_name)
            except Exception as e:
                log_warning(f"[ExperimentTracker] 记录实验失败: {e}")
            
        except Exception as e:
            import traceback
            sdata['train_status'] = 'error'
            sdata['train_error'] = str(e)
            sdata['train_error_stack'] = traceback.format_exc()
            sdata['train_config'] = config
            log_warning(f"[Web] 训练失败: {e}")
    
    sdata['train_status'] = 'running'
    sdata['train_error'] = None
    sdata['pipeline_result'] = None
    sdata['modeling_result'] = None
    sdata['train_events'] = []
    sdata['train_live_results'] = []
    
    thread = threading.Thread(target=train_task)
    thread.start()
    
    return jsonify({'success': True, 'status': 'running'})


@app.route('/api/model/status', methods=['GET'])
def api_model_status():
    """查询训练状态"""
    sdata = get_session()
    status = sdata.get('train_status', 'idle')
    error = sdata.get('train_error')
    progress = sdata.get('train_progress')
    return jsonify({'success': True, 'status': status, 'error': error, 'progress': progress})


@app.route('/api/model/train-events', methods=['GET'])
def api_model_train_events():
    """获取训练实时事件流（since_id 支持增量获取）"""
    sdata = get_session()
    since_id = request.args.get('since_id', type=int, default=-1)
    events = sdata.get('train_events', [])
    new_events = [e for e in events if e.get('id', 0) > since_id]
    return jsonify({'success': True, 'events': new_events, 'latest_id': events[-1].get('id', -1) if events else -1})


@app.route('/api/model/live-results', methods=['GET'])
def api_model_live_results():
    """获取已完成的模型实时结果（迷你排行榜数据）"""
    sdata = get_session()
    live_results = sdata.get('train_live_results', [])
    return jsonify({'success': True, 'results': live_results})


@app.route('/api/model/result', methods=['GET'])
def api_model_result():
    """获取训练结果"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    model_result = sdata.get('model_result', {})
    
    # 多目标模式：直接返回汇总
    if model_result and model_result.get('multi_target'):
        return jsonify(clean_for_json({
            'success': True,
            'result': {
                'multi_target': True,
                'targets': model_result.get('targets', []),
                'leaderboards': model_result.get('leaderboards', []),
                'total_time': model_result.get('total_time', 0),
            }
        }))
    
    if mresult is None:
        return jsonify({'success': False, 'error': '尚未完成训练'}), 400
    
    # 排行榜
    leaderboard = []
    if mresult.leaderboard is not None and not mresult.leaderboard.empty:
        leaderboard = mresult.leaderboard.to_dict('records')
    
    # 决策报告
    decision = {}
    if mresult.decision_report:
        dr = mresult.decision_report
        decision = {
            'mode': dr.mode.value if hasattr(dr.mode, 'value') else str(dr.mode),
            'mode_description': dr.mode_description,
            'recommended_model': dr.recommended_model,
            'recommended_name': dr.recommended_name,
            'recommendation_reason': dr.recommendation_reason,
            'confidence': dr.confidence,
            'risks': dr.risks,
            'scenario_advice': dr.scenario_advice,
            'override_options': dr.override_options,
        }
        if dr.comparison_table is not None:
            decision['comparison_table'] = dr.comparison_table.to_dict('records')
        # 传递各模型的五维评分供前端可视化
        if dr.scores:
            decision['model_scores'] = [
                {
                    'model_key': s.model_key,
                    'model_name': s.model_name,
                    'accuracy_score': round(s.accuracy_score, 1),
                    'speed_score': round(s.speed_score, 1),
                    'stability_score': round(s.stability_score, 1),
                    'simplicity_score': round(s.simplicity_score, 1),
                    'generalization_score': round(s.generalization_score, 1),
                    'composite_score': round(s.composite_score, 1),
                    'primary_score': round(s.primary_score, 4) if isinstance(s.primary_score, (int, float)) else s.primary_score,
                    'train_time': round(s.train_time, 2) if isinstance(s.train_time, (int, float)) else s.train_time,
                }
                for s in dr.scores
            ]
    
    # 采样报告
    sampling = {}
    if mresult.sampling_report:
        sr = mresult.sampling_report
        sampling = {
            'original_n': sr.original_n,
            'sampled_n': sr.sampled_n,
            'sample_ratio': sr.sample_ratio,
            'strategy': sr.strategy,
            'distribution_preservation': sr.distribution_preservation,
        }
    
    # 预处理信息（移除不可序列化的对象如 encoder/feature_selector/label_encoder）
    preprocessing = {k: v for k, v in mresult.preprocessing_info.items() 
                     if k not in ('encoder', 'scaler', 'feature_selector', 'label_encoder', 'autoencoder')}
    
    # 编码报告
    encoding_report = []
    if mresult.encoding_report is not None and not mresult.encoding_report.empty:
        encoding_report = mresult.encoding_report.to_dict('records')
    
    # 特征选择报告
    feature_selection_report = []
    if mresult.feature_selection_report is not None and not mresult.feature_selection_report.empty:
        feature_selection_report = mresult.feature_selection_report.to_dict('records')
    
    # 融合权重
    ensemble_weights = mresult.ensemble_result.get('weights') if mresult.ensemble_result else None
    
    # Permutation Importance
    permutation_importance = []
    if mresult.permutation_importance is not None and not mresult.permutation_importance.empty:
        permutation_importance = mresult.permutation_importance.head(30).to_dict('records')
    
    # 超参优化结果
    optimized_params = mresult.optimized_params or {}
    optimization_history = mresult.optimization_history or {}
    
    return jsonify(clean_for_json({
        'success': True,
        'result': {
            'task_type': mresult.task_type.value,
            'best_model_key': mresult.best_model_key,
            'leaderboard': leaderboard,
            'decision': decision,
            'sampling': sampling,
            'preprocessing': preprocessing,
            'encoding_report': encoding_report,
            'feature_selection_report': feature_selection_report,
            'ensemble_weights': ensemble_weights,
            'permutation_importance': permutation_importance,
            'pseudo_label_report': mresult.pseudo_label_report,
            'train_time': mresult.train_time,
            'optimized_params': optimized_params,
            'optimization_history': optimization_history,
        }
    }))


@app.route('/api/model/hyperopt-history', methods=['GET'])
def api_hyperopt_history():
    """获取超参数优化历史（每次trial的参数和分数）"""
    sdata = get_session()
    history = sdata.get('model_result', {}).get('optimization_history')
    optimized_params = sdata.get('model_result', {}).get('optimized_params')
    
    if history is None:
        return jsonify({'success': False, 'error': '未启用超参优化或尚未完成训练'}), 400
    
    # 构建模型列表（包含历史记录、最优参数和搜索空间）
    models = []
    task_type = sdata.get('model_result', {}).get('task_type')
    for model_key, trials in history.items():
        model_info = {
            'model_key': model_key,
            'model_name': model_key,
            'trial_count': len(trials),
            'best_params': optimized_params.get(model_key, {}) if optimized_params else {},
            'trials': trials,
            'search_space': {},
        }
        # 从 ModelLibrary 获取搜索空间定义
        if task_type:
            try:
                task_enum = TaskType(task_type) if isinstance(task_type, str) else task_type
                spec = ModelLibrary.get_models(task_type=task_enum).get(model_key)
                if spec and spec.hyperparam_space:
                    model_info['search_space'] = spec.hyperparam_space
            except Exception:
                pass
        models.append(model_info)
    
    return jsonify(clean_for_json({
        'success': True,
        'models': models,
        'total_trials': sum(len(m['trials']) for m in models),
    }))


@app.route('/api/model/tune', methods=['POST'])
def api_model_tune():
    """手动调参重新评估模型"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    df = sdata.get('df')
    config = sdata.get('train_config', {})
    pipeline_result = sdata.get('pipeline_result')
    
    if mresult is None or df is None:
        return jsonify({'success': False, 'error': '尚未完成训练'}), 400
    
    data = request.get_json() or {}
    model_key = data.get('model_key')
    params = data.get('params', {})
    
    if not model_key:
        return jsonify({'success': False, 'error': '请指定模型'}), 400
    
    # 获取任务类型和预处理信息
    task_type = mresult.task_type
    preproc = mresult.preprocessing_info or {}
    encoder = preproc.get('encoder')
    feature_selector = preproc.get('feature_selector')
    autoencoder = preproc.get('autoencoder')
    label_encoder = preproc.get('label_encoder')
    
    # 获取特征和目标
    target_col = (pipeline_result.target_col if pipeline_result else None) or config.get('target_col')
    if target_col and target_col in df.columns:
        X_raw = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X_raw = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # 应用预处理
    X_proc = X_raw.copy()
    if encoder is not None:
        X_proc = encoder.transform(X_proc)
    if feature_selector is not None:
        X_proc = feature_selector.transform(X_proc)
    if autoencoder is not None:
        try:
            X_proc = pd.DataFrame(autoencoder.transform(X_proc), index=X_proc.index)
        except Exception as e:
            log_warning(f"[Tune] autoencoder transform failed: {e}")
    
    # 标签编码（分类任务）
    if task_type == TaskType.CLASSIFICATION and label_encoder is not None:
        y = pd.Series(label_encoder.transform(y.astype(str)), index=y.index, name=y.name)
    
    try:
        # 创建模型
        model = ModelLibrary.create_model(model_key, task_type, **params)
        
        # CV 评估
        n_splits = config.get('n_splits', 5)
        cv = CrossValidator(n_splits=n_splits, random_state=42, verbose=False)
        result = cv.cross_validate(model, X_proc, y, task_type)
        
        # 提取主指标
        from core.modeling_engine import TaskTypeDetector
        primary_metric = TaskTypeDetector.get_primary_metric(task_type)
        score = result.mean_scores.get(primary_metric, 0)
        cv_scores = result.fold_scores.get(primary_metric, [])
        # 回归且主指标是 rmse 时，score 需要反转以便统一"越大越好"
        is_regression_rmse = False
        if task_type.value == 'regression' and primary_metric == 'rmse':
            score = 1.0 / (score + 1e-6)
            cv_scores = [1.0 / (s + 1e-6) for s in cv_scores]
            is_regression_rmse = True
        
        # 构建每折诊断
        fold_diagnostics = []
        for fold_idx in range(len(cv_scores)):
            fd = {
                'fold': fold_idx + 1,
                'score': cv_scores[fold_idx] if not is_regression_rmse else 1.0 / (result.fold_scores.get(primary_metric, [])[fold_idx] + 1e-6),
                'metrics': {k: result.fold_scores.get(k, [])[fold_idx] if fold_idx < len(result.fold_scores.get(k, [])) else 0 
                           for k in result.fold_scores.keys()}
            }
            fold_diagnostics.append(fd)
        
        # 混淆矩阵 / 残差统计
        diagnostics = {}
        if task_type.value == 'classification' and result.oof_pred is not None and y is not None:
            try:
                from sklearn.metrics import confusion_matrix
                y_true = y.values if hasattr(y, 'values') else np.array(y)
                cm = confusion_matrix(y_true, result.oof_pred)
                diagnostics['confusion_matrix'] = cm.tolist()
                diagnostics['confusion_matrix_labels'] = sorted(set(y_true))
            except Exception:
                pass
        elif task_type.value == 'regression' and result.oof_pred is not None and y is not None:
            try:
                y_true = y.values if hasattr(y, 'values') else np.array(y)
                residuals = y_true - result.oof_pred
                diagnostics['residuals'] = {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'max_abs': float(np.max(np.abs(residuals))),
                    'samples': min(100, len(residuals)),
                    'values': residuals[:min(100, len(residuals))].tolist()
                }
            except Exception:
                pass
        
        # 特征重要性
        fi = None
        if result.feature_importance is not None:
            try:
                fi = result.feature_importance.head(15).to_dict('records')
            except Exception:
                pass
        
        return jsonify(clean_for_json({
            'success': True,
            'model_key': model_key,
            'params': params,
            'score': score,
            'cv_scores': cv_scores,
            'scores': result.mean_scores,
            'std_scores': result.std_scores,
            'train_time': round(result.train_time, 2),
            'fold_diagnostics': fold_diagnostics,
            'diagnostics': diagnostics,
            'feature_importance': fi,
            'task_type': task_type.value,
        }))
    except Exception as e:
        import traceback
        log_warning(f"[Tune] {model_key} evaluation failed: {e}")
        return jsonify({'success': False, 'error': str(e), 'detail': traceback.format_exc()}), 500


@app.route('/api/model/export', methods=['POST'])
def api_model_export():
    """导出模型摘要与可复用代码"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    if mresult is None:
        return jsonify({'success': False, 'error': '尚未完成训练'}), 400
    
    data = request.get_json() or {}
    model_key = data.get('model_key') or mresult.best_model_key
    format_type = data.get('format', 'both')
    
    if not model_key:
        return jsonify({'success': False, 'error': '请指定模型'}), 400
    
    # 获取模型规格
    task_type = mresult.task_type
    models = ModelLibrary.get_models(task_type)
    model_spec = models.get(model_key)
    if not model_spec:
        return jsonify({'success': False, 'error': f'Unknown model: {model_key}'}), 400
    
    # 获取最终参数
    params = {}
    if mresult.optimized_params and model_key in mresult.optimized_params:
        params = mresult.optimized_params[model_key]
    else:
        params = dict(model_spec.default_params)
    
    # 编码报告
    encoding_report = None
    if mresult.encoding_report is not None and not mresult.encoding_report.empty:
        encoding_report = mresult.encoding_report.to_dict('records')
    
    # 特征选择报告
    feature_selection_report = None
    if mresult.feature_selection_report is not None and not mresult.feature_selection_report.empty:
        feature_selection_report = mresult.feature_selection_report.to_dict('records')
    
    # CV 结果
    cv_result = None
    for r in mresult.cv_results:
        if r.model_key == model_key:
            cv_result = r
            break
    
    try:
        from core.code_generator import generate_model_export
        result = generate_model_export(
            model_key=model_key,
            task_type=task_type,
            model_spec=model_spec,
            params=params,
            preprocessing_info=mresult.preprocessing_info,
            encoding_report=encoding_report,
            feature_selection_report=feature_selection_report,
            cv_result=cv_result,
        )
        
        resp = {
            'success': True,
            'model_key': model_key,
            'model_name': result['model_name'],
        }
        if format_type in ('summary', 'both'):
            resp['summary'] = result['summary']
        if format_type in ('code', 'both'):
            resp['code'] = result['code']
        return jsonify(resp)
    except Exception as e:
        import traceback
        log_warning(f"[Export] failed: {e}")
        return jsonify({'success': False, 'error': str(e), 'detail': traceback.format_exc()}), 500


@app.route('/api/experiments', methods=['GET'])
def api_experiments():
    """获取实验历史"""
    from core.experiment_tracker import list_experiments
    task_type = request.args.get('task_type')
    limit = int(request.args.get('limit', 50))
    rows = list_experiments(limit=limit, task_type=task_type)
    return jsonify({'success': True, 'experiments': rows})


@app.route('/api/experiments/<int:exp_id>', methods=['GET'])
def api_experiment_detail(exp_id):
    """获取单个实验详情"""
    from core.experiment_tracker import get_experiment
    exp = get_experiment(exp_id)
    if exp is None:
        return jsonify({'success': False, 'error': '实验不存在'}), 404
    return jsonify({'success': True, 'experiment': exp})


@app.route('/api/experiments/compare', methods=['POST'])
def api_experiments_compare():
    """对比多个实验"""
    from core.experiment_tracker import compare_experiments
    data = request.get_json() or {}
    ids = data.get('ids', [])
    if not ids:
        return jsonify({'success': False, 'error': '请指定实验ID'}), 400
    results = compare_experiments(ids)
    return jsonify({'success': True, 'comparison': results})


@app.route('/api/experiments/<int:exp_id>', methods=['DELETE'])
def api_experiment_delete(exp_id):
    """删除实验"""
    from core.experiment_tracker import delete_experiment
    ok = delete_experiment(exp_id)
    if ok:
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '删除失败'}), 500


@app.route('/api/model/fairness', methods=['POST'])
def api_model_fairness():
    """模型公平性分析"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    df = sdata.get('df')
    config = sdata.get('train_config', {})
    
    if mresult is None or df is None:
        return jsonify({'success': False, 'error': '尚未完成训练'}), 400
    
    data = request.get_json() or {}
    model_key = data.get('model_key')
    sensitive_attr = data.get('sensitive_attr')
    
    # 找到对应模型的 CVResult
    target_cv = None
    for cv in mresult.cv_results:
        if cv.model_key == model_key:
            target_cv = cv
            break
    
    if target_cv is None or not target_cv.fitted_models:
        return jsonify({'success': False, 'error': f'模型 {model_key} 无训练结果'}), 400
    
    model = target_cv.fitted_models[-1]
    target_col = config.get('target_col')
    
    # 获取特征和目标（原始数据）
    # 优先使用 pipeline 自动识别的 target_col（可能和用户传入的不同）
    pipeline_result = sdata.get('pipeline_result')
    target_col = (pipeline_result.target_col if pipeline_result else None) or config.get('target_col')
    
    # 聚类任务没有目标列，所有列都是特征
    is_clustering = (mresult.task_type == TaskType.CLUSTERING) or \
                    (isinstance(mresult.task_type, str) and mresult.task_type == 'clustering')
    
    if is_clustering:
        X_raw = df.copy()
        y = None
    elif target_col and target_col in df.columns:
        X_raw = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        # 未指定目标列时，默认最后一列为目标（与训练逻辑一致）
        X_raw = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # 应用与训练时相同的预处理，得到模型期望的特征
    preproc = mresult.preprocessing_info or {}
    encoder = preproc.get('encoder')
    feature_selector = preproc.get('feature_selector')
    label_encoder = preproc.get('label_encoder')
    
    X_proc = X_raw.copy()
    if encoder is not None:
        X_proc = encoder.transform(X_proc)
    if feature_selector is not None:
        X_proc = feature_selector.transform(X_proc)
    # 应用训练时的 autoencoder 降维（如果启用）
    autoencoder = preproc.get('autoencoder')
    log_info(f"[DEBUG fairness] autoencoder={type(autoencoder).__name__ if autoencoder else None}, X_proc_before={X_proc.shape}")
    if autoencoder is not None:
        try:
            X_proc = pd.DataFrame(autoencoder.transform(X_proc), index=X_proc.index)
            log_info(f"[DEBUG fairness] X_proc_after={X_proc.shape}")
        except Exception as e:
            log_warning(f"[DEBUG fairness] autoencoder transform failed: {e}")
            raise
    
    # 用预处理后的数据做预测
    y_pred = model.predict(X_proc)
    # 如果训练时对目标列做了编码，预测结果需要逆变换回原始标签
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred.astype(int))
    
    try:
        from core.fairness import FairnessEngine
        engine = FairnessEngine(fairness_threshold=0.05)
        
        # 公平性分析在原始数据上进行（敏感属性在原始列中）
        # 如果用户未指定敏感属性，在原始特征中自动检测
        created_temp_attr = False
        if sensitive_attr is None:
            candidates = engine.detect_sensitive_attributes(X_raw)
            log_info(f"[DEBUG fairness] detect_sensitive_attributes candidates={candidates}, X_raw dtypes={X_raw.dtypes.to_dict()}")
            if not candidates:
                # Fallback: 将第一个数值列按中位数二值化作为合成敏感属性
                numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
                log_info(f"[DEBUG fairness] numeric_cols={numeric_cols}")
                if numeric_cols:
                    col = numeric_cols[0]
                    binned_name = f"{col}_binned"
                    if binned_name not in X_raw.columns:
                        X_raw = X_raw.copy()
                        X_raw[binned_name] = (X_raw[col] > X_raw[col].median()).astype(int).astype(str)
                        created_temp_attr = True
                        candidates = [binned_name]
                        log_info(f"[DEBUG fairness] created temp attr {binned_name}")
            if not candidates:
                return jsonify({'success': False, 'error': '未检测到候选敏感属性（需要低基数分类列，如性别、年龄组等）'}), 400
            sensitive_attr = candidates[0]
        
        if sensitive_attr not in df.columns and sensitive_attr not in X_raw.columns:
            return jsonify({'success': False, 'error': f'敏感属性 "{sensitive_attr}" 不在数据中'}), 400
        
        # 构造只包含敏感属性的 DataFrame 传给 analyze（已传入 y_pred 避免再次预测）
        if sensitive_attr in df.columns:
            X_fair = df[[sensitive_attr]].copy()
        else:
            X_fair = X_raw[[sensitive_attr]].copy()
        
        report = engine.analyze(
            model, X_fair, y,
            y_pred=y_pred,
            sensitive_attr=sensitive_attr,
            task_type=mresult.task_type.value
        )
        
        if report is None:
            return jsonify({'success': False, 'error': '公平性分析失败'}), 400
        
        result = {
            'model_key': report.model_key,
            'sensitive_attr': report.sensitive_attr,
            'demographic_parity_diff': report.demographic_parity_diff,
            'demographic_parity_ratio': report.demographic_parity_ratio,
            'equalized_odds_diff': report.equalized_odds_diff,
            'equalized_odds_ratio': report.equalized_odds_ratio,
            'fpr_diff': report.fpr_diff,
            'fnr_diff': report.fnr_diff,
            'selection_rate_diff': report.selection_rate_diff,
            'group_metrics': report.group_metrics,
            'is_fair': report.is_fair,
            'recommendations': report.recommendations,
            'analysis_time': report.analysis_time,
        }
        
        return jsonify(clean_for_json({'success': True, 'result': result}))
    except Exception as e:
        return api_error_response(str(e), detail=traceback.format_exc())


@app.route('/api/model/explain', methods=['POST'])
def api_model_explain():
    """模型可解释性分析 (SHAP / LIME)"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    df = sdata.get('df')
    config = sdata.get('train_config', {})
    
    if mresult is None or df is None:
        return jsonify({'success': False, 'error': '尚未完成训练'}), 400
    
    data = request.get_json() or {}
    model_key = data.get('model_key')
    instance_index = data.get('instance_index', 0)
    
    # 找到对应模型的 CVResult
    target_cv = None
    for cv in mresult.cv_results:
        if cv.model_key == model_key:
            target_cv = cv
            break
    
    if target_cv is None or not target_cv.fitted_models:
        return jsonify({'success': False, 'error': f'模型 {model_key} 无训练结果'}), 400
    
    model = target_cv.fitted_models[-1]
    
    # 获取特征数据，并应用与训练时相同的预处理
    # 优先使用 pipeline 自动识别的 target_col（可能和用户传入的不同）
    pipeline_result = sdata.get('pipeline_result')
    target_col = (pipeline_result.target_col if pipeline_result else None) or config.get('target_col')
    
    # 聚类任务没有目标列，所有列都是特征
    is_clustering = (mresult.task_type == TaskType.CLUSTERING) or \
                    (isinstance(mresult.task_type, str) and mresult.task_type == 'clustering')
    
    if is_clustering:
        X = df.copy()
    elif target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        # 未指定目标列时，默认最后一列为目标（与训练逻辑一致）
        X = df.iloc[:, :-1]
    
    # 应用训练时的编码和特征选择
    preproc = mresult.preprocessing_info or {}
    encoder = preproc.get('encoder')
    feature_selector = preproc.get('feature_selector')
    if encoder is not None:
        X = encoder.transform(X)
    if feature_selector is not None:
        X = feature_selector.transform(X)
    # 应用训练时的 autoencoder 降维（如果启用）
    autoencoder = preproc.get('autoencoder')
    if autoencoder is not None:
        X = pd.DataFrame(autoencoder.transform(X), index=X.index)
    # 确保无缺失值
    X = X.fillna(0)
    
    try:
        from core.explainability import ExplainabilityEngine
        engine = ExplainabilityEngine()
        
        # 全局解释
        global_exp = engine.explain_model(
            model, X, y=None, model_key=model_key,
            task_type=mresult.task_type,
            feature_names=list(X.columns)
        )
        
        # 局部解释（单样本）
        instance_exp = engine.explain_instance(
            model, X, instance_index=min(instance_index, len(X)-1),
            model_key=model_key, task_type=mresult.task_type,
            feature_names=list(X.columns)
        )
        
        # 序列化结果
        result = {
            'model_key': model_key,
            'model_name': target_cv.model_name,
            'method': global_exp.method,
            'explanation_time': global_exp.explanation_time,
            'global_importance': [],
            'instance': {
                'index': instance_exp.get('instance_index', 0),
                'prediction': instance_exp.get('prediction', {}),
                'top_positive': instance_exp.get('top_positive', []),
                'top_negative': instance_exp.get('top_negative', []),
                'lime': instance_exp.get('lime', {}),
            },
        }
        
        if global_exp.global_importance is not None:
            result['global_importance'] = global_exp.global_importance.head(20).to_dict('records')
        
        return jsonify(clean_for_json({'success': True, 'result': result}))
    except Exception as e:
        return api_error_response(str(e), detail=traceback.format_exc())


@app.route('/api/model/explain/shap', methods=['POST'])
def api_model_explain_shap():
    """SHAP model explainability"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    df = sdata.get('df')
    config = sdata.get('train_config', {})
    if mresult is None or df is None:
        return jsonify({'success': False, 'error': 'Training not completed'}), 400
    data = request.get_json() or {}
    model_key = data.get('model_key')
    target_cv = None
    for cv in mresult.cv_results:
        if cv.model_key == model_key:
            target_cv = cv
            break
    if target_cv is None or not target_cv.fitted_models:
        return jsonify({'success': False, 'error': f'Model {model_key} not found'}), 400
    model = target_cv.fitted_models[-1]
    pipeline_result = sdata.get('pipeline_result')
    target_col = (pipeline_result.target_col if pipeline_result else None) or config.get('target_col')
    is_clustering = (mresult.task_type == TaskType.CLUSTERING) or \
                    (isinstance(mresult.task_type, str) and mresult.task_type == 'clustering')
    if is_clustering:
        X = df.copy()
    elif target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df.iloc[:, :-1]
    preproc = mresult.preprocessing_info or {}
    encoder = preproc.get('encoder')
    feature_selector = preproc.get('feature_selector')
    if encoder is not None:
        X = encoder.transform(X)
    if feature_selector is not None:
        X = feature_selector.transform(X)
    autoencoder = preproc.get('autoencoder')
    if autoencoder is not None:
        X = pd.DataFrame(autoencoder.transform(X), index=X.index)
    X = X.fillna(0)
    try:
        from core.shap_explainer import explain_model
        result = explain_model(model, X, task_type=mresult.task_type.value if hasattr(mresult.task_type, 'value') else str(mresult.task_type), sample_size=100)
        return jsonify({'success': True, 'shap': result})
    except Exception as e:
        log_error(f'[SHAP] {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# 可视化 API
# =============================================================================

@app.route('/api/visualization/generate', methods=['POST'])
def api_visualization_generate():
    """生成图表"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    df = sdata.get('df')
    
    # 多目标模式：取第一个目标的结果用于可视化
    multi_results = sdata.get('multi_target_results')
    if multi_results and mresult is None:
        first = list(multi_results.values())[0]
        mresult = first.get('modeling_result')
    
    if mresult is None or df is None:
        return jsonify({'success': False, 'error': '数据或模型结果不存在'}), 400
    
    data = request.get_json() or {}
    chart_type = data.get('chart_type', 'leaderboard')
    
    try:
        fig = None
        
        if chart_type == 'leaderboard':
            mv = ModelVisualizer()
            if mresult.leaderboard is not None and not mresult.leaderboard.empty:
                fig = mv.plot_leaderboard(mresult.leaderboard)
            else:
                return jsonify({'success': False, 'error': '当前结果无排行榜数据（所有模型训练失败）'}), 400
        elif chart_type == 'feature_importance':
            mv = ModelVisualizer()
            if mresult.feature_importance is not None and not mresult.feature_importance.empty:
                fig = mv.plot_feature_importance(mresult)
            else:
                return jsonify({'success': False, 'error': '当前结果无特征重要性数据（聚类任务或无可用重要性）'}), 400
        elif chart_type == 'decision_radar':
            if mresult.decision_report:
                ev = EvaluationVisualizer()
                fig = ev.plot_radar_comparison(mresult.decision_report)
            else:
                return jsonify({'success': False, 'error': '当前结果无决策报告（聚类任务或未启用自动评估）'}), 400
        elif chart_type == 'score_breakdown':
            if mresult.decision_report:
                ev = EvaluationVisualizer()
                fig = ev.plot_score_breakdown(mresult.decision_report)
            else:
                return jsonify({'success': False, 'error': '当前结果无决策报告（聚类任务或未启用自动评估）'}), 400
        elif chart_type == 'cv_boxplot':
            mv = ModelVisualizer()
            # 聚类任务没有 fold_scores，无法绘制 CV 箱线图
            has_fold_scores = any(
                r.fold_scores for r in mresult.cv_results
            )
            if has_fold_scores:
                fig = mv.plot_cv_boxplot(mresult.cv_results)
            else:
                return jsonify({'success': False, 'error': '当前结果无交叉验证分数（聚类任务或不支持CV的模型）'}), 400
        elif chart_type == 'correlation':
            dv = DataVisualizer()
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] >= 2:
                fig = dv.plot_correlation_heatmap(df)
            else:
                return jsonify({'success': False, 'error': '数值列不足，无法绘制相关性热力图'}), 400
        elif chart_type == 'missing':
            dv = DataVisualizer()
            fig = dv.plot_missing_values(df)
        elif chart_type == 'hyperopt_history':
            from core.visualization import plot_optimization_history
            history = sdata.get('model_result', {}).get('optimization_history')
            if history:
                path = plot_optimization_history(history, save_path='hyperopt_history.png')
                if path:
                    import matplotlib.pyplot as plt
                    fig = plt.imread(path)
                    # 重新加载为Figure对象以便统一输出
                    fig2, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(fig)
                    ax.axis('off')
                    fig = fig2
            else:
                return jsonify({'success': False, 'error': '未启用超参优化或尚无历史记录'}), 400
        elif chart_type == 'reward_curve':
            from core.visualization import plot_reward_curve
            # 查找 RL 历史
            rl_history = None
            history = sdata.get('model_result', {}).get('optimization_history', {})
            for model_key, trials in history.items():
                if trials and any('reward' in t for t in trials):
                    rl_history = trials
                    break
            if rl_history:
                path = plot_reward_curve(rl_history, save_path='reward_curve.png')
                if path:
                    import matplotlib.pyplot as plt
                    fig = plt.imread(path)
                    fig2, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(fig)
                    ax.axis('off')
                    fig = fig2
            else:
                return jsonify({'success': False, 'error': '未找到 RL 优化历史'}), 400
        elif chart_type == 'autoencoder':
            from core.visualization import plot_autoencoder_results
            # 尝试从 session 中获取 AutoEncoder 结果
            ae_result = sdata.get('ae_result')
            if ae_result:
                path = plot_autoencoder_results(
                    ae_result['X_orig'], ae_result['X_recon'],
                    ae_result.get('encoded'),
                    save_path='autoencoder_results.png'
                )
                if path:
                    import matplotlib.pyplot as plt
                    fig = plt.imread(path)
                    fig2, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(fig)
                    ax.axis('off')
                    fig = fig2
            else:
                return jsonify({'success': False, 'error': '未找到 AutoEncoder 结果（请在训练时选择降维为自编码器）'}), 400
        
        if fig is None:
            return jsonify({'success': False, 'error': '图表生成失败'}), 500
        
        # 转为 base64
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({'success': True, 'image': f'data:image/png;base64,{img_base64}'})
        
    except Exception as e:
        return api_error_response(str(e))


# =============================================================================
# 预测 API
# =============================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """上传测试集进行预测"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    if mresult is None:
        return jsonify({'success': False, 'error': '请先完成训练'}), 400
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '未找到文件'}), 400
    
    file = request.files['file']
    ext = os.path.splitext(file.filename)[1].lower()
    
    try:
        if ext == '.csv':
            test_df = pd.read_csv(file)
        elif ext in ('.xls', '.xlsx'):
            test_df = pd.read_excel(file)
        else:
            return jsonify({'success': False, 'error': '仅支持 CSV/Excel'}), 400
        
        # 使用最优模型预测
        if mresult.task_type == TaskType.CLUSTERING:
            # 聚类任务：用最佳模型的 predict() 分配聚类标签
            if not mresult.best_cv_result or not mresult.best_cv_result.fitted_models:
                return jsonify({'success': False, 'error': '无可用预测模型'}), 400
            
            best_model = mresult.best_cv_result.fitted_models[-1]
            if not hasattr(best_model, 'predict'):
                return jsonify({'success': False, 'error': f'聚类模型 {mresult.best_cv_result.model_name} 不支持对新数据预测'}), 400
            
            # 对新数据做与训练时相同的预处理
            X_pred = test_df.copy()
            prep = mresult.preprocessing_info or {}
            encoder = prep.get('encoder')
            scaler = prep.get('scaler')
            
            if encoder is not None:
                try:
                    X_pred = encoder.transform(X_pred)
                except Exception as e:
                    log_warning(f"[Web] 聚类预测编码失败: {e}，尝试 fit_transform")
                    try:
                        X_pred = encoder.fit_transform(X_pred)
                    except Exception as e2:
                        return jsonify({'success': False, 'error': f'编码失败: {e2}'}), 400
            
            if scaler is not None:
                try:
                    X_pred = pd.DataFrame(
                        scaler.transform(X_pred),
                        columns=X_pred.columns,
                        index=X_pred.index
                    )
                except Exception as e:
                    return jsonify({'success': False, 'error': f'标准化失败: {e}'}), 400
            
            predictions = best_model.predict(X_pred)
            
        elif mresult.ensemble_result and 'test' in mresult.ensemble_result:
            predictions = mresult.ensemble_result['test']
        else:
            return jsonify({'success': False, 'error': '无可用预测模型'}), 400
        
        # 构造结果
        if predictions is None:
            return jsonify({'success': False, 'error': '模型未产生预测结果，可能是测试集为空或模型不支持预测'}), 400
        result_df = test_df.copy()
        result_df['prediction'] = predictions[:len(result_df)]
        
        # 保存结果到会话
        result_path = UPLOAD_DIR / f"pred_{uuid.uuid4().hex}.csv"
        result_df.to_csv(result_path, index=False)
        sdata['prediction_path'] = str(result_path)
        
        return jsonify({
            'success': True,
            'preview': result_df.head(20).to_dict('records'),
            'shape': result_df.shape,
            'download_url': f'/api/export/result?sid={session.get("sid")}'
        })
        
    except Exception as e:
        return api_error_response(str(e))


@app.route('/api/export/result', methods=['GET'])
def api_export_result():
    """导出预测结果"""
    sdata = get_session()
    pred_path = sdata.get('prediction_path')
    if pred_path and os.path.exists(pred_path):
        return send_file(pred_path, as_attachment=True, download_name='predictions.csv')
    return jsonify({'success': False, 'error': '无预测结果'}), 404


# =============================================================================
# 大模型智能分析 API
# =============================================================================

@app.route('/api/llm/config', methods=['GET'])
def api_llm_config():
    """获取 LLM 默认配置"""
    return jsonify({
        'success': True,
        'providers': get_default_configs(),
    })


@app.route('/api/llm/analyze', methods=['POST'])
def api_llm_analyze():
    """触发大模型分析"""
    sdata = get_session()
    data = request.get_json() or {}
    
    analysis_type = data.get('analysis_type')
    if analysis_type not in ('eda', 'result', 'error'):
        return jsonify({'success': False, 'error': '分析类型必须是 eda/result/error 之一'}), 400
    
    # 校验分析类型是否可用
    if analysis_type == 'eda' and not sdata.get('df_info'):
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    if analysis_type == 'result' and not sdata.get('model_result'):
        return jsonify({'success': False, 'error': '请先完成训练'}), 400
    if analysis_type == 'error' and not sdata.get('train_error'):
        return jsonify({'success': False, 'error': '当前无错误记录'}), 400
    
    # LLM 配置
    llm_config = LLMConfig(
        provider=data.get('provider', 'openai'),
        base_url=data.get('base_url', 'https://api.openai.com/v1'),
        api_key=data.get('api_key', ''),
        model_name=data.get('model_name', 'gpt-4o'),
    )
    sdata['llm_config'] = {
        'provider': llm_config.provider,
        'base_url': llm_config.base_url,
        'model_name': llm_config.model_name,
    }
    
    def analyze_task():
        try:
            analyzer = LLMAnalyzer(llm_config)
            result = analyzer.analyze(analysis_type, sdata)
            sdata['llm_analysis_status'] = 'done'
            sdata['llm_analysis_result'] = result
            sdata['llm_analysis_error'] = None
            log_info(f"[Web] LLM 分析完成: {analysis_type}")
        except Exception as e:
            sdata['llm_analysis_status'] = 'error'
            sdata['llm_analysis_result'] = None
            sdata['llm_analysis_error'] = str(e)
            log_warning(f"[Web] LLM 分析失败: {e}")
    
    sdata['llm_analysis_status'] = 'running'
    sdata['llm_analysis_result'] = None
    sdata['llm_analysis_error'] = None
    
    thread = threading.Thread(target=analyze_task)
    thread.start()
    
    return jsonify({'success': True, 'status': 'running'})


@app.route('/api/llm/status', methods=['GET'])
def api_llm_status():
    """查询 LLM 分析状态"""
    sdata = get_session()
    status = sdata.get('llm_analysis_status', 'idle')
    error = sdata.get('llm_analysis_error')
    result = sdata.get('llm_analysis_result')
    return jsonify({
        'success': True,
        'status': status,
        'error': error,
        'result': result,
    })


@app.route('/api/ollama/models', methods=['GET'])
def api_ollama_models():
    """获取 Ollama 已安装的本地模型列表"""
    base_url = request.args.get('base_url', 'http://localhost:11434')
    try:
        import requests
        # Ollama 原生 API: /api/tags
        url = f"{base_url.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get('models', [])
        # 提取模型名称，按名称排序
        model_names = sorted([m.get('name', '') for m in models if m.get('name')])
        return jsonify({'success': True, 'models': model_names, 'source': 'ollama'})
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'error': '无法连接到 Ollama，请确认服务已启动'}), 503
    except Exception as e:
        return jsonify({'success': False, 'error': f'获取模型列表失败: {str(e)}'}), 500


# 支持的本地模型文件扩展名
LOCAL_MODEL_EXTENSIONS = {'.gguf', '.bin', '.safetensors'}

@app.route('/api/local-models/scan', methods=['POST'])
def api_scan_local_models():
    """扫描本地文件夹中的模型文件"""
    data = request.get_json() or {}
    path = data.get('path', '')
    
    if not path:
        return jsonify({'success': False, 'error': '请提供文件夹路径'}), 400
    
    try:
        target = Path(path)
        if not target.exists():
            return jsonify({'success': False, 'error': f'路径不存在: {path}'}), 400
        if not target.is_dir():
            return jsonify({'success': False, 'error': f'不是文件夹: {path}'}), 400
        
        models = []
        for item in target.iterdir():
            if item.is_file() and item.suffix.lower() in LOCAL_MODEL_EXTENSIONS:
                models.append({
                    'name': item.stem,           # 文件名（无扩展名）
                    'filename': item.name,       # 完整文件名
                    'size_mb': round(item.stat().st_size / (1024 * 1024), 1),
                    'path': str(item),
                })
        
        # 按文件名排序
        models.sort(key=lambda x: x['name'].lower())
        
        return jsonify({
            'success': True,
            'models': models,
            'source': 'folder',
            'path': str(target),
            'count': len(models),
        })
    except PermissionError:
        return jsonify({'success': False, 'error': f'无权限访问: {path}'}), 403
    except Exception as e:
        return jsonify({'success': False, 'error': f'扫描失败: {str(e)}'}), 500


# =============================================================================
# 报表设计 API
# =============================================================================

@app.route('/api/report/fields', methods=['GET'])
def api_report_fields():
    """获取可用字段列表"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    
    fields = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype == 'object':
            field_type = 'text'
        elif 'datetime' in dtype:
            field_type = 'datetime'
        elif 'bool' in dtype:
            field_type = 'boolean'
        else:
            field_type = 'numeric'
        
        fields.append({
            'name': col,
            'type': field_type,
            'dtype': dtype,
        })
    
    return jsonify({'success': True, 'fields': fields})


@app.route('/api/report/preview', methods=['POST'])
def api_report_preview():
    """预览报表"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    
    data = request.get_json() or {}
    config_dict = data.get('config', {})
    
    try:
        config = _parse_report_config(config_dict)
        img_bytes = ReportEngine.preview(df, config)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return jsonify({'success': True, 'image': f'data:image/png;base64,{img_base64}'})
    except Exception as e:
        log_warning(f"[Web] 报表预览失败: {e}")
        return api_error_response(str(e))


@app.route('/api/report/export', methods=['POST'])
def api_report_export():
    """导出报表"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    
    data = request.get_json() or {}
    config_dict = data.get('config', {})
    fmt = data.get('format', 'excel')
    
    if fmt not in ('excel', 'pdf'):
        return jsonify({'success': False, 'error': '格式必须是 excel 或 pdf'}), 400
    
    try:
        config = _parse_report_config(config_dict)
        ext = '.xlsx' if fmt == 'excel' else '.pdf'
        export_path = UPLOAD_DIR / f"report_{uuid.uuid4().hex}{ext}"
        ReportEngine.export(df, config, fmt, str(export_path))
        
        return send_file(
            str(export_path),
            as_attachment=True,
            download_name=f"{config.title or '报表'}{ext}",
        )
    except Exception as e:
        log_warning(f"[Web] 报表导出失败: {e}")
        return api_error_response(str(e))


@app.route('/api/report/chart/preview', methods=['POST'])
def api_report_chart_preview():
    """单独预览图表"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': '请先上传数据'}), 400
    
    data = request.get_json() or {}
    chart_dict = data.get('chart', {})
    
    try:
        from extensions.report_engine import ChartConfig, ChartBuilder
        chart_cfg = ChartConfig(
            chart_type=chart_dict.get('chart_type', 'bar'),
            x_field=chart_dict.get('x_field', ''),
            y_field=chart_dict.get('y_field', ''),
            group_field=chart_dict.get('group_field', ''),
            agg=chart_dict.get('agg', 'sum'),
            title=chart_dict.get('title', '图表'),
            color_scheme=chart_dict.get('color_scheme', 'default'),
            show_values=chart_dict.get('show_values', True),
            top_n=chart_dict.get('top_n', 0),
        )
        img_bytes = ChartBuilder.build_to_bytes(df, chart_cfg)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return jsonify({'success': True, 'image': f'data:image/png;base64,{img_base64}'})
    except Exception as e:
        log_warning(f"[Web] 图表预览失败: {e}")
        return api_error_response(str(e))


@app.route('/api/report/save', methods=['POST'])
def api_report_save():
    """保存报表设计"""
    sdata = get_session()
    data = request.get_json() or {}
    sdata['report_config'] = data.get('config')
    return jsonify({'success': True})


def _parse_report_config(config_dict: Dict) -> ReportConfig:
    """从字典解析报表配置"""
    from extensions.report_engine import ChartConfig
    
    mode = config_dict.get('mode', 'pivot')
    title = config_dict.get('title', '报表')
    styles = config_dict.get('styles', {})
    
    # 解析图表配置
    charts = []
    for chart_dict in config_dict.get('charts', []):
        charts.append(ChartConfig(
            chart_type=chart_dict.get('chart_type', 'bar'),
            x_field=chart_dict.get('x_field', ''),
            y_field=chart_dict.get('y_field', ''),
            group_field=chart_dict.get('group_field', ''),
            agg=chart_dict.get('agg', 'sum'),
            title=chart_dict.get('title', '图表'),
            color_scheme=chart_dict.get('color_scheme', 'default'),
            show_values=chart_dict.get('show_values', True),
            top_n=chart_dict.get('top_n', 0),
        ))
    
    if mode == 'pivot':
        pivot_dict = config_dict.get('pivot', {})
        pivot = PivotConfig(
            row_fields=pivot_dict.get('row_fields', []),
            col_fields=pivot_dict.get('col_fields', []),
            value_fields=pivot_dict.get('value_fields', []),
            aggregations=pivot_dict.get('aggregations', {}),
            filters=pivot_dict.get('filters', {}),
        )
        return ReportConfig(mode=mode, title=title, pivot=pivot, charts=charts, styles=styles)
    else:
        cells = []
        for cell_dict in config_dict.get('cells', []):
            cells.append(CellConfig(
                row=cell_dict.get('row', 0),
                col=cell_dict.get('col', 0),
                row_span=cell_dict.get('row_span', 1),
                col_span=cell_dict.get('col_span', 1),
                type=cell_dict.get('type', 'text'),
                value=cell_dict.get('value', ''),
                bound_field=cell_dict.get('bound_field'),
                agg=cell_dict.get('agg', 'sum'),
                format=cell_dict.get('format', ''),
                style=cell_dict.get('style', {}),
            ))
        return ReportConfig(
            mode=mode, title=title, cells=cells, charts=charts,
            row_count=config_dict.get('row_count', 12),
            col_count=config_dict.get('col_count', 8),
            styles=styles,
        )


# =============================================================================
# 运行日志 API
# =============================================================================

@app.route('/api/logs', methods=['GET'])
def api_logs():
    """获取运行日志"""
    level = request.args.get('level', 'ALL')
    limit = int(request.args.get('limit', 200))
    offset = int(request.args.get('offset', 0))
    
    store = get_log_store()
    logs, total = store.get_logs(level=level, limit=limit, offset=offset)
    stats = store.get_stats()
    
    return jsonify({
        'success': True,
        'logs': logs,
        'total': total,
        'stats': stats,
    })


@app.route('/api/logs/stats', methods=['GET'])
def api_logs_stats():
    """获取日志统计"""
    return jsonify({
        'success': True,
        'stats': get_log_store().get_stats(),
    })


@app.route('/api/logs/clear', methods=['POST'])
def api_logs_clear():
    """清空日志"""
    get_log_store().clear()
    return jsonify({'success': True})


# =============================================================================
# 依赖管理 API
# =============================================================================

@app.route('/api/dependencies', methods=['GET'])
def api_dependencies():
    """获取可选依赖状态列表"""
    deps = get_missing_dependencies()
    missing_count = sum(1 for d in deps if not d['installed'])
    return jsonify({
        'success': True,
        'dependencies': deps,
        'missing_count': missing_count,
    })


@app.route('/api/dependencies/install', methods=['POST'])
def api_dependencies_install():
    """安装指定依赖到指定目录"""
    data = request.get_json() or {}
    package_key = data.get('package')
    install_all = data.get('install_all', False)
    target_dir_str = data.get('target_dir')
    
    # 解析目标目录
    if target_dir_str:
        target_dir = Path(target_dir_str).resolve()
        # 安全检查：防止路径遍历到项目根目录之外
        try:
            target_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            return jsonify({'success': False, 'error': '目标路径必须在项目目录内'}), 400
    else:
        target_dir = DEFAULT_THIRD_PARTY_DIR
    
    if install_all:
        results = install_all_missing(target_dir)
        if any(r[0] for r in results.values()):
            try:
                ModelLibrary.refresh()
            except Exception as e:
                app.logger.warning(f'模型库刷新失败: {e}')
        return jsonify({
            'success': all(r[0] for r in results.values()),
            'results': {k: {'success': v[0], 'stdout': v[1], 'stderr': v[2]} for k, v in results.items()},
        })
    
    if not package_key or package_key not in OPTIONAL_DEPENDENCIES:
        return jsonify({'success': False, 'error': f'未知依赖: {package_key}'}), 400
    
    success, stdout, stderr = install_dependency(package_key, target_dir)
    if success:
        try:
            ModelLibrary.refresh()
        except Exception as e:
            app.logger.warning(f'模型库刷新失败: {e}')
    return jsonify({
        'success': success,
        'package': package_key,
        'target_dir': str(target_dir),
        'stdout': stdout[-2000:] if stdout else '',
        'stderr': stderr[-2000:] if stderr else '',
    })


# =============================================================================
# 决策辅助 API
# =============================================================================

@app.route('/api/decision/modes', methods=['GET'])
def api_decision_modes():
    """获取所有决策模式说明"""
    modes = [
        {'key': 'balanced', 'name': '平衡模式', 'desc': '综合考虑精度、速度、稳定性，适合大多数场景'},
        {'key': 'accuracy_first', 'name': '精度优先', 'desc': '选择CV分数最高的模型，适合比赛打榜'},
        {'key': 'speed_first', 'name': '速度优先', 'desc': '选择训练快、推断快的模型，适合实时应用'},
        {'key': 'stability_first', 'name': '稳定性优先', 'desc': '选择CV方差最小的模型，适合生产环境'},
        {'key': 'simplicity_first', 'name': '简单优先', 'desc': '选择复杂度低的模型，防止过拟合'},
    ]
    return jsonify({'success': True, 'modes': modes})


@app.route('/api/model/options', methods=['GET'])
def api_model_options():
    """获取可用模型列表（包含超参空间）"""
    from core.modeling_engine import ModelLibrary
    
    models = []
    for task in ['classification', 'regression', 'clustering']:
        task_models = ModelLibrary.get_models(task_type=TaskType(task))
        for key, spec in task_models.items():
            # 序列化超参空间
            hyperparam_space = {}
            for param_name, param_values in spec.hyperparam_space.items():
                if isinstance(param_values, dict) and param_values.get('type') == 'float':
                    hyperparam_space[param_name] = {
                        'type': 'float',
                        'low': param_values.get('low'),
                        'high': param_values.get('high'),
                        'scale': param_values.get('scale', 'linear'),
                    }
                else:
                    hyperparam_space[param_name] = {
                        'type': 'categorical',
                        'choices': param_values,
                    }
            models.append({
                'key': key,
                'name': spec.name,
                'task_type': task,
                'category': spec.category,
                'default_params': spec.default_params,
                'hyperparam_space': hyperparam_space,
            })
    
    return jsonify({'success': True, 'models': models})


# =============================================================================
# 设置管理
# =============================================================================

SETTINGS_FILE = PROJECT_ROOT / 'workspace' / 'config' / 'settings.json'
SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

_default_settings = {
    'api_base_url': 'http://localhost:5000',
    'http_proxy': '',
    'request_timeout': 30,
    'sidebar_visible': True,
    'sidebar_width': 220,
    'theme_mode': 'light',
    'primary_color': '#2E86AB',
    'font_size': 'medium',
    'language': 'zh-CN',
    'log_level': 'INFO',
    'auto_save': True,
    'performance_mode': False,
}

def _load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 合并默认值，确保新字段存在
            merged = dict(_default_settings)
            merged.update(data)
            return merged
        except Exception:
            pass
    return dict(_default_settings)

def _save_settings(data):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        log_warning(f"保存设置失败: {e}")
        return False

@app.route('/api/settings', methods=['GET'])
def api_settings_get():
    """获取全部设置"""
    return jsonify({'success': True, 'settings': _load_settings()})

@app.route('/api/settings', methods=['POST'])
def api_settings_post():
    """保存设置"""
    data = request.get_json() or {}
    current = _load_settings()
    current.update(data)
    _save_settings(current)
    return jsonify({'success': True, 'message': 'Settings saved'})

@app.route('/api/settings/backup', methods=['POST'])
def api_settings_backup():
    """导出配置备份"""
    settings = _load_settings()
    # 同时包含数据集元信息
    datasets_meta = []
    sdata = get_session()
    dm = sdata.get('data_module')
    if dm and hasattr(dm, 'datasets'):
        datasets_meta = [{'name': d.name, 'shape': list(d.df.shape) if hasattr(d, 'df') else None}
                         for d in dm.datasets]
    backup = {
        'version': '1.0',
        'exported_at': pd.Timestamp.now().isoformat(),
        'settings': settings,
        'datasets_meta': datasets_meta,
    }
    buf = io.BytesIO(json.dumps(backup, ensure_ascii=False, indent=2).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='application/json',
                     as_attachment=True,
                     download_name=f"smartchart_backup_{pd.Timestamp.now().strftime('%Y%m%d')}.json")

@app.route('/api/settings/restore', methods=['POST'])
def api_settings_restore():
    """导入配置恢复"""
    try:
        data = request.get_json()
        if not data or 'settings' not in data:
            return jsonify({'success': False, 'error': 'Invalid backup format'}), 400
        _save_settings(data['settings'])
        return jsonify({'success': True, 'message': 'Settings restored'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# API Key 管理（简单实现：存储在 settings 中）
_api_key_file = PROJECT_ROOT / 'workspace' / 'config' / '.api_key'

def _get_api_key():
    if _api_key_file.exists():
        try:
            return _api_key_file.read_text(encoding='utf-8').strip()
        except Exception:
            pass
    return None

def _set_api_key(key):
    try:
        _api_key_file.parent.mkdir(parents=True, exist_ok=True)
        _api_key_file.write_text(key, encoding='utf-8')
        return True
    except Exception as e:
        log_warning(f"保存 API Key 失败: {e}")
        return False

def _generate_api_key():
    return 'sk-' + base64.urlsafe_b64encode(os.urandom(24)).decode('ascii').rstrip('=')

@app.route('/api/settings/api-key', methods=['GET'])
def api_settings_api_key_get():
    """获取开发者 API Key"""
    key = _get_api_key()
    return jsonify({'success': True, 'api_key': key})

@app.route('/api/settings/api-key/regenerate', methods=['POST'])
def api_settings_api_key_regenerate():
    """生成/重置 API Key"""
    data = request.get_json(silent=True) or {}
    if data.get('revoke'):
        if _api_key_file.exists():
            _api_key_file.unlink()
        return jsonify({'success': True, 'api_key': None})
    new_key = _generate_api_key()
    _set_api_key(new_key)
    return jsonify({'success': True, 'api_key': new_key})


# =============================================================================
# 全局异常处理
# =============================================================================

@app.errorhandler(Exception)
def handle_global_exception(e):
    """捕获所有未处理异常，记录完整 traceback 并返回标准错误响应"""
    if isinstance(e, HTTPException):
        return e
    endpoint = request.endpoint or 'unknown'
    detail = traceback.format_exc()
    msg = f"[{endpoint}] {type(e).__name__}: {str(e)}"
    app.logger.error(f"{msg}\n{detail}")
    log_error(msg, category="API")
    return jsonify({
        'success': False,
        'error': str(e),
        'endpoint': endpoint,
        'detail': detail,
    }), 500


def _wrap_api_handlers():
    """为所有 api_ 开头的视图函数自动添加异常处理和日志记录"""
    for endpoint, view_func in list(app.view_functions.items()):
        if not endpoint.startswith('api_'):
            continue
        original = view_func
        @wraps(original)
        def wrapper(*args, __original=original, __endpoint=endpoint, **kwargs):
            try:
                return __original(*args, **kwargs)
            except Exception as e:
                detail = traceback.format_exc()
                msg = f"[{__endpoint}] {type(e).__name__}: {str(e)}"
                app.logger.error(f"{msg}\n{detail}")
                log_error(msg, category="API")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'endpoint': __endpoint,
                    'detail': detail,
                }), 500
        app.view_functions[endpoint] = wrapper


# =============================================================================
# 模型快照与回滚
# =============================================================================

@app.route('/api/model/snapshot', methods=['POST'])
def api_model_snapshot():
    """Save current best model snapshot"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    if not mresult or not mresult.cv_results:
        return jsonify({'success': False, 'error': 'No trained model'}), 400
    data = request.get_json() or {}
    model_key = data.get('model_key')
    if not model_key:
        return jsonify({'success': False, 'error': 'model_key required'}), 400
    target_cv = None
    for cv in mresult.cv_results:
        if cv.model_key == model_key:
            target_cv = cv
            break
    if not target_cv or not target_cv.fitted_models:
        return jsonify({'success': False, 'error': 'Model not found'}), 400
    try:
        from core.model_versioning import save_snapshot
        from core.experiment_tracker import list_experiments
        exp_rows = list_experiments(limit=1)
        exp_id = exp_rows[0]['id'] if exp_rows else 0
        sid = save_snapshot(exp_id, model_key, target_cv.fitted_models[-1], metadata={'score': getattr(target_cv, 'cv_score', None)})
        return jsonify({'success': True, 'snapshot_id': sid})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/rollback', methods=['POST'])
def api_model_rollback():
    """Rollback to a saved snapshot"""
    sdata = get_session()
    data = request.get_json() or {}
    snapshot_id = data.get('snapshot_id')
    if not snapshot_id:
        return jsonify({'success': False, 'error': 'snapshot_id required'}), 400
    try:
        from core.model_versioning import load_snapshot
        model = load_snapshot(snapshot_id)
        if model is None:
            return jsonify({'success': False, 'error': 'Snapshot not found'}), 404
        sdata['rollback_model'] = model
        return jsonify({'success': True, 'message': 'Rollback model loaded into session'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/incremental', methods=['POST'])
def api_model_incremental():
    """Incremental learning with new data batch"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    if not mresult or not mresult.cv_results:
        return jsonify({'success': False, 'error': 'No trained model'}), 400
    data = request.get_json() or {}
    model_key = data.get('model_key')
    if not model_key:
        return jsonify({'success': False, 'error': 'model_key required'}), 400
    target_cv = None
    for cv in mresult.cv_results:
        if cv.model_key == model_key:
            target_cv = cv
            break
    if not target_cv or not target_cv.fitted_models:
        return jsonify({'success': False, 'error': 'Model not found'}), 400
    model = target_cv.fitted_models[-1]
    try:
        from core.incremental_learning import partial_fit_model, supports_incremental
        if not supports_incremental(model_key):
            return jsonify({'success': False, 'error': f'Model {model_key} does not support incremental learning'}), 400
        df = sdata.get('df')
        config = sdata.get('train_config', {})
        pipeline_result = sdata.get('pipeline_result')
        target_col = (pipeline_result.target_col if pipeline_result else None) or config.get('target_col')
        is_clustering = (mresult.task_type == TaskType.CLUSTERING) or \
                        (isinstance(mresult.task_type, str) and mresult.task_type == 'clustering')
        if is_clustering:
            return jsonify({'success': False, 'error': 'Clustering does not support incremental learning'}), 400
        if not target_col or target_col not in df.columns:
            return jsonify({'success': False, 'error': 'Target column not set'}), 400
        X = df.drop(columns=[target_col])
        y = df[target_col]
        preproc = mresult.preprocessing_info or {}
        encoder = preproc.get('encoder')
        feature_selector = preproc.get('feature_selector')
        if encoder is not None:
            X = encoder.transform(X)
        if feature_selector is not None:
            X = feature_selector.transform(X)
        X = X.fillna(0)
        classes = list(y.unique()) if hasattr(y, 'unique') else None
        updated = partial_fit_model(model, X, y, classes=classes)
        target_cv.fitted_models[-1] = updated
        return jsonify({'success': True, 'message': 'Model updated via incremental learning'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Report & Deploy & Robustness & TimeSeries
# =============================================================================

@app.route('/api/model/report', methods=['POST'])
def api_model_report():
    """Generate modeling report"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    data = request.get_json() or {}
    fmt = data.get('format', 'html')
    try:
        from core.report_generator import generate_html_report, generate_word_report
        model_dict = {}
        if mresult:
            model_dict = {
                'task_type': str(mresult.task_type) if hasattr(mresult, 'task_type') else '',
                'best_model': getattr(mresult, 'best_model', {}),
                'leaderboard': mresult.leaderboard.to_dict('records') if hasattr(mresult.leaderboard, 'to_dict') else [],
            }
        df = sdata.get('df')
        data_info = {'shape': list(df.shape), 'columns': list(df.columns)} if df is not None else None
        if fmt == 'html':
            html = generate_html_report(model_dict, data_info)
            return jsonify({'success': True, 'html': html})
        else:
            path = generate_word_report(model_dict, data_info, output_path='report.docx')
            return jsonify({'success': True, 'path': path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/deploy', methods=['POST'])
def api_model_deploy():
    """Deploy model as REST API package"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    if not mresult or not mresult.cv_results:
        return jsonify({'success': False, 'error': 'No trained model'}), 400
    data = request.get_json() or {}
    model_key = data.get('model_key')
    target_cv = None
    for cv in mresult.cv_results:
        if cv.model_key == model_key:
            target_cv = cv
            break
    if not target_cv or not target_cv.fitted_models:
        return jsonify({'success': False, 'error': 'Model not found'}), 400
    try:
        from core.model_deployer import generate_deploy_package
        files = generate_deploy_package(target_cv.fitted_models[-1], output_dir='deploy_package')
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/robustness', methods=['POST'])
def api_model_robustness():
    """Run robustness tests"""
    sdata = get_session()
    mresult = sdata.get('modeling_result')
    df = sdata.get('df')
    if not mresult or not df is not None:
        return jsonify({'success': False, 'error': 'No model or data'}), 400
    data = request.get_json() or {}
    model_key = data.get('model_key')
    target_cv = None
    for cv in mresult.cv_results:
        if cv.model_key == model_key:
            target_cv = cv
            break
    if not target_cv or not target_cv.fitted_models:
        return jsonify({'success': False, 'error': 'Model not found'}), 400
    config = sdata.get('train_config', {})
    pipeline_result = sdata.get('pipeline_result')
    target_col = (pipeline_result.target_col if pipeline_result else None) or config.get('target_col')
    is_clustering = (mresult.task_type == TaskType.CLUSTERING) or \
                    (isinstance(mresult.task_type, str) and mresult.task_type == 'clustering')
    if is_clustering:
        return jsonify({'success': False, 'error': 'Clustering not supported'}), 400
    if not target_col or target_col not in df.columns:
        return jsonify({'success': False, 'error': 'Target not set'}), 400
    X = df.drop(columns=[target_col])
    y = df[target_col]
    preproc = mresult.preprocessing_info or {}
    encoder = preproc.get('encoder')
    feature_selector = preproc.get('feature_selector')
    if encoder is not None:
        X = encoder.transform(X)
    if feature_selector is not None:
        X = feature_selector.transform(X)
    X = X.fillna(0)
    try:
        from core.robustness_tester import evaluate_robustness
        task_type = 'classification' if mresult.task_type == TaskType.CLASSIFICATION else 'regression'
        result = evaluate_robustness(target_cv.fitted_models[-1], X, y, task_type=task_type)
        return jsonify({'success': True, 'robustness': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/timeseries', methods=['POST'])
def api_data_timeseries():
    """Time series analysis"""
    sdata = get_session()
    df = sdata.get('df')
    if df is None:
        return jsonify({'success': False, 'error': 'No data'}), 400
    data = request.get_json() or {}
    col = data.get('column')
    if not col or col not in df.columns:
        return jsonify({'success': False, 'error': 'Column not found'}), 400
    try:
        from core.time_series_analysis import analyze_time_series, prophet_forecast
        series = df[col]
        if not pd.api.types.is_datetime64_any_dtype(series.index):
            series.index = pd.to_datetime(df.index)
        result = analyze_time_series(series, freq=data.get('freq'), period=data.get('period'))
        if data.get('prophet'):
            forecast = prophet_forecast(series, periods=data.get('periods', 10))
            if forecast:
                result['prophet'] = forecast
        return jsonify({'success': True, 'analysis': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


_wrap_api_handlers()
log_info("API 异常处理自动包装完成", category="System")


# =============================================================================
# 主入口
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SmartChart Web Service Starting")
    print("URL: http://localhost:5000")
    print("=" * 60)
    # Windows 下 reloader 与后台训练线程冲突，禁用自动重载
    use_reloader = os.name != 'nt'
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=use_reloader)
