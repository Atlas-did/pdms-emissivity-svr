#!/usr/bin/env python3
"""
PDMS-TMM-SVR Workflow Diagram Generator
符合SCI期刊要求的高质量流程图
使用Graphviz生成PDF/PNG/SVG格式
"""

from graphviz import Digraph
import os


def create_tmm_svr_workflow(output_dir='./output', format='pdf'):
    """
    创建TMM-SVR系统完整流程图
    """

    # 创建有向图
    dot = Digraph(
        name='TMM_SVR_Workflow',
        comment='PDMS Thin Film Emissivity Prediction Workflow',
        format=format,
        engine='dot'  # 使用dot布局引擎
    )

    # 设置全局属性
    dot.attr(
        rankdir='TB',  # 从上到下布局
        bgcolor='white',
        fontname='Arial',
        fontsize='12',
        margin='20',
        nodesep='0.5',
        ranksep='0.8',
        splines='ortho',  # 正交线条
        concentrate='true'
    )

    # 定义颜色方案（色盲友好）
    colors = {
        'input': '#E8F5E9',  # 浅绿 - 输入/配置
        'physics': '#E3F2FD',  # 浅蓝 - 物理计算
        'validation': '#FFEBEE',  # 浅红 - 验证/检查
        'ml': '#FFF3E0',  # 浅橙 - 机器学习
        'decision': '#F3E5F5',  # 浅紫 - 决策
        'output': '#FFFDE7',  # 浅黄 - 输出
        'cache': '#E0F7FA',  # 青绿 - 缓存
        'error': '#FFCDD2',  # 红色 - 错误处理
    }

    # 定义节点样式
    def add_node(name, label, color_type, shape='box', style='filled,rounded',
                 fontsize='10', width='2.5', height='0.6'):
        """添加标准节点"""
        dot.node(
            name,
            label=label,
            shape=shape,
            style=style,
            fillcolor=colors[color_type],
            fontname='Arial',
            fontsize=fontsize,
            width=width,
            height=height,
            color='#333333',
            penwidth='1.2'
        )

    def add_decision(name, label, color_type='decision'):
        """添加决策节点（菱形）"""
        dot.node(
            name,
            label=label,
            shape='diamond',
            style='filled',
            fillcolor=colors[color_type],
            fontname='Arial',
            fontsize='9',
            width='1.8',
            height='1.0',
            color='#333333',
            penwidth='1.2'
        )

    # ==================== Phase 1: 数据准备 ====================
    dot.attr(rank='same')
    with dot.subgraph(name='cluster_phase1') as c:
        c.attr(
            label='Phase 1: Data Preparation',
            style='rounded',
            bgcolor='#F5F5F5',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        add_node('config', 'Load Configuration\\n(config.json)', 'input')
        add_node('material', 'Load Optical Constants\\n(pdms_nk.xlsx, sio2_nk.xlsx)', 'input')
        add_node('param_grid', 'Build Parameter Grid\\n(λ×d_PDMS×d_SiO₂)', 'input')

        # 数据质量检查
        add_decision('data_check', 'Physical\\nValidity?')
        add_node('data_clean', 'Data Cleaning\\nk < -0.01 → 0\\n-0.01 ≤ k < 0: keep', 'validation')
        add_node('data_error', 'ERROR:\\nInvalid data', 'error')

        c.edge('material', 'data_check')
        c.edge('data_check', 'data_clean', label='Yes', color='#2E7D32', penwidth='2')
        c.edge('data_check', 'data_error', label='No', style='dashed', color='#C62828')
        c.edge('config', 'param_grid')
        c.edge('data_clean', 'param_grid', style='invis')

    # ==================== Phase 2: TMM物理仿真（含缓存） ====================
    with dot.subgraph(name='cluster_phase2') as c:
        c.attr(
            label='Phase 2: TMM Physical Simulation',
            style='rounded',
            bgcolor='#E8EAF6',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        # 缓存查询
        add_decision('cache_check', 'TTL Cache\\nHit?', 'cache')
        add_node('cache_return', 'Return Cached\\nSpectrum', 'cache')
        add_node('tmm_compute', 'TMM Computation\\n(parallel via joblib)', 'physics')

        # TMM详细步骤
        add_node('phase_thick', 'Calculate Phase Thickness\\nδⱼ = 2π(nⱼ\'+iκⱼ)dⱼ/λ', 'physics')
        add_node('char_matrix', 'Characteristic Matrix\\nMⱼ = [[cosδⱼ, i·sinδⱼ/ηⱼ], [i·ηⱼ·sinδⱼ, cosδⱼ]]', 'physics')
        add_node('sys_matrix', 'System Matrix\\nM = M_N·M_{N-1}·...·M_1\\n(ordered: incident → substrate)', 'physics')
        add_node('rt_coeff',
                 'Reflection/Transmission\\nr = (η₀M₁₁ + η₀ηₛM₁₂ - M₂₁ - ηₛM₂₂) / (...)\\nR = |r|², T = |t|²',
                 'physics')
        add_node('cache_store', 'Store in Cache\\n(TTL=180s)', 'cache')

        # 连接
        c.edge('param_grid', 'cache_check')
        c.edge('cache_check', 'cache_return', label='Hit', color='#2E7D32', penwidth='2')
        c.edge('cache_check', 'tmm_compute', label='Miss', color='#1565C0', penwidth='2')
        c.edge('tmm_compute', 'phase_thick')
        c.edge('phase_thick', 'char_matrix')
        c.edge('char_matrix', 'sys_matrix')
        c.edge('sys_matrix', 'rt_coeff')
        c.edge('rt_coeff', 'cache_store')
        c.edge('cache_store', 'cache_return', style='invis')

    # ==================== Phase 3: 数据验证 ====================
    with dot.subgraph(name='cluster_phase3') as c:
        c.attr(
            label='Phase 3: Physical Validation',
            style='rounded',
            bgcolor='#FFEBEE',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        add_node('emissivity', 'Calculate Emissivity\\nε = 1 - R - T\\n(if opaque: ε = 1 - R)', 'physics')
        add_decision('energy_check', '|R+T+ε-1|\\n< 10⁻³?', 'validation')
        add_node('energy_pass', 'Energy Conservation\\nValidated', 'validation')
        add_node('energy_fail', 'FLAG: Energy Violation\\nExclude from training', 'error')
        add_node('save_valid', 'Save Valid Data\\n(tmm_emissivity_data.csv)', 'output')

        c.edge('cache_return', 'emissivity')
        c.edge('emissivity', 'energy_check')
        c.edge('energy_check', 'energy_pass', label='Yes', color='#2E7D32', penwidth='2')
        c.edge('energy_check', 'energy_fail', label='No', style='dashed', color='#C62828')
        c.edge('energy_pass', 'save_valid')

    # ==================== Phase 4: 特征工程 ====================
    with dot.subgraph(name='cluster_phase4') as c:
        c.attr(
            label='Phase 4: Feature Engineering',
            style='rounded',
            bgcolor='#FFF3E0',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        add_node('base_features', 'Base Features\\nX = [λ, d_SiO₂, d_PDMS]', 'ml')
        add_node('phys_features', 'Physical Features\\nφ = [sinδ, cosδ, 1/λ, d/λ]', 'ml')
        add_node('interaction', 'Interaction Features\\n[sinδ₁·sinδ₂, ...]', 'ml')

        c.edge('save_valid', 'base_features')
        c.edge('base_features', 'phys_features')
        c.edge('phys_features', 'interaction')

    # ==================== Phase 5: 数据划分 ====================
    with dot.subgraph(name='cluster_phase5') as c:
        c.attr(
            label='Phase 5: Data Splitting (Prevent Leakage)',
            style='rounded',
            bgcolor='#F3E5F5',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        add_decision('group_split', 'Thickness-Grouped\\nSplit?', 'decision')
        add_node('grouped', 'GroupShuffleSplit\\nPrevent spectral correlation leakage', 'ml')
        add_node('random_split', 'Random Split\\n(Not recommended)', 'error')
        add_node('three_way', 'Three-Way Split\\nTrain 70% | Val 15% | Test 15%', 'ml')
        add_node('scaler_fit', 'StandardScaler fit\\n(ONLY on training set)', 'ml')
        add_node('scaler_transform', 'StandardScaler transform\\n(train & test)', 'ml')

        c.edge('interaction', 'group_split')
        c.edge('group_split', 'grouped', label='Yes', color='#2E7D32', penwidth='2')
        c.edge('group_split', 'random_split', label='No', style='dashed', color='#C62828')
        c.edge('grouped', 'three_way')
        c.edge('three_way', 'scaler_fit')
        c.edge('scaler_fit', 'scaler_transform')

    # ==================== Phase 6: 超参优化 ====================
    with dot.subgraph(name='cluster_phase6') as c:
        c.attr(
            label='Phase 6: Hyperparameter Optimization',
            style='rounded',
            bgcolor='#E0F7FA',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        add_node('search_baseline', 'Baseline Search\\nGridSearchCV / RandomizedSearchCV', 'ml')
        add_node('search_halton', 'Proposed: Halton Sequence\\n(quasi-random, bases 2,3 for C,γ)', 'ml')
        add_node('search_bayesian', 'Optional: Bayesian Opt.\\n(Optuna, refine best region)', 'ml')

        add_decision('converge_check', 'Converged?', 'decision')
        add_node('best_params', 'Best Parameters\\n(C*, γ*, ε*)', 'output')

        c.edge('scaler_transform', 'search_baseline', style='dashed')
        c.edge('scaler_transform', 'search_halton')
        c.edge('search_halton', 'search_bayesian', label='refine', style='dashed')
        c.edge('search_bayesian', 'converge_check')
        c.edge('search_baseline', 'converge_check', style='invis')
        c.edge('converge_check', 'search_halton', label='No', style='dashed')
        c.edge('converge_check', 'best_params', label='Yes', color='#2E7D32', penwidth='2')

    # ==================== Phase 7: SVR训练 ====================
    with dot.subgraph(name='cluster_phase7') as c:
        c.attr(
            label='Phase 7: SVR Training',
            style='rounded',
            bgcolor='#FFF8E1',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        add_node('svr_train', 'SVR Training\\nRBF Kernel: K(x,x\') = exp(-γ||x-x\'||²)', 'ml')
        add_node('cv_eval', 'Cross-Validation\\n5-fold thickness-grouped CV', 'validation')
        add_node('save_model', 'Save Best Model\\n(model.pkl)', 'output')

        c.edge('best_params', 'svr_train')
        c.edge('svr_train', 'cv_eval')
        c.edge('cv_eval', 'save_model')

    # ==================== Phase 8: 模型评估 ====================
    with dot.subgraph(name='cluster_phase8') as c:
        c.attr(
            label='Phase 8: Model Evaluation',
            style='rounded',
            bgcolor='#E8F5E9',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        add_node('metrics', 'Performance Metrics\\nR², RMSE, MAE, SV count', 'validation')
        add_node('extrap_test', 'Extrapolation Test\\nUnseen thickness: [750,1000] nm', 'validation')
        add_node('speed_test', 'Speed Benchmark\\nSpeedup = t_TMM / t_SVR', 'validation')
        add_node('uncertainty', 'Optional: Uncertainty\\nBayesian SVR / MC Dropout', 'validation')

        c.edge('save_model', 'metrics')
        c.edge('metrics', 'extrap_test')
        c.edge('extrap_test', 'speed_test')
        c.edge('speed_test', 'uncertainty', style='dashed')

    # ==================== Phase 9: 多区域策略 ====================
    with dot.subgraph(name='cluster_phase9') as c:
        c.attr(
            label='Phase 9: Multi-Region Strategy (if needed)',
            style='rounded',
            bgcolor='#FCE4EC',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        add_decision('need_multi', 'Single Region\\nR² < 0.98?', 'decision')
        add_node('region_split', 'RegionSplitter\\nλ-based partitioning', 'ml')
        add_node('overlap', '5% Overlap Region\\nΩₖ^overlap = [λ_{k+1}^min, λₖ^max]', 'ml')
        add_node('weighted', 'Weighted Blending\\nwₖ = d_{k+1}/(dₖ + d_{k+1})', 'ml')
        add_node('router', 'MultiRegionModelRouter\\nPhase 1-3 routing', 'ml')

        c.edge('metrics', 'need_multi', constraint='false')
        c.edge('need_multi', 'region_split', label='Yes', color='#1565C0', penwidth='2')
        c.edge('need_multi', 'final_output', label='No', style='invis')
        c.edge('region_split', 'overlap')
        c.edge('overlap', 'weighted')
        c.edge('weighted', 'router')

    # ==================== Phase 10: 输出与验证 ====================
    with dot.subgraph(name='cluster_phase10') as c:
        c.attr(
            label='Phase 10: Output & Experimental Validation',
            style='rounded',
            bgcolor='#E8EAF6',
            color='#666666',
            fontname='Arial Bold',
            fontsize='11',
            margin='15'
        )

        add_node('final_output', 'Predicted Emissivity\\nSpectrum ε(λ,d)', 'output')

        # 椭偏验证（独立分支）
        add_node('ellipsometry', 'Ellipsometry Data\\n(Ψ, Δ) → (n, k)', 'input')
        add_node('ellip_process', 'Extract (n,k)\\nCubic spline interpolation', 'physics')
        add_node('ellip_tmm', 'TMM with measured (n,k)', 'physics')
        add_node('ellip_compare', 'Compare: Predicted vs. Measured ε', 'validation')

        c.edge('router', 'final_output', style='invis')
        c.edge('uncertainty', 'final_output')
        c.edge('ellipsometry', 'ellip_process')
        c.edge('ellip_process', 'ellip_tmm')
        c.edge('ellip_tmm', 'ellip_compare')
        c.edge('final_output', 'ellip_compare', style='dashed',
               label='validate', color='#666666')

    # ==================== 关键假设注释 ====================
    # 添加注释节点
    dot.attr(rank='sink')
    with dot.subgraph(name='cluster_notes') as c:
        c.attr(
            label='Key Assumptions & Parameters',
            style='rounded,dashed',
            bgcolor='#FAFAFA',
            color='#999999',
            fontname='Arial Italic',
            fontsize='9',
            margin='10'
        )

        note_text = (
            'Assumptions: (1) SiO₂ substrate semi-infinite (d >> λ); '
            '(2) PDMS isotropic, homogeneous; (3) Coherent superposition; '
            '(4) Normal incidence (θ = 0°)\\n'
            'Parameters: λ ∈ [2,14] μm, d_PDMS ∈ [100,1000] nm, '
            'Energy tol = 10⁻³, Cache TTL = 180s, Overlap = 5%'
        )

        c.node('notes', note_text,
               shape='note',
               fillcolor='#FFFDE7',
               fontname='Arial',
               fontsize='8',
               width='8',
               height='1.2',
               color='#666666')

    # 连接注释（不可见）
    dot.edge('final_output', 'notes', style='invis')

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'tmm_svr_workflow_revised.{format}')
    dot.render(output_path, cleanup=True)
    print(f"流程图已保存: {output_path}")

    # 同时保存DOT源文件
    dot_source_path = os.path.join(output_dir, 'tmm_svr_workflow_revised.dot')
    with open(dot_source_path, 'w') as f:
        f.write(dot.source)
    print(f"DOT源文件已保存: {dot_source_path}")

    return dot


def create_simplified_workflow(output_dir='./output', format='pdf'):
    """
    创建简化版流程图（用于正文，单栏宽度）
    """

    dot = Digraph(
        name='TMM_SVR_Simplified',
        comment='Simplified workflow for main text',
        format=format,
        engine='dot'
    )

    dot.attr(
        rankdir='LR',  # 从左到右，适合单栏
        bgcolor='white',
        fontname='Arial',
        fontsize='11',
        size='8,10',  # 页面尺寸限制
        ratio='compress',
        nodesep='0.3',
        ranksep='0.5'
    )

    colors = {
        'input': '#E8F5E9',
        'physics': '#E3F2FD',
        'validation': '#FFEBEE',
        'ml': '#FFF3E0',
        'output': '#FFFDE7',
    }

    def add_compact_node(name, label, color_type):
        dot.node(
            name,
            label=label,
            shape='box',
            style='filled,rounded',
            fillcolor=colors[color_type],
            fontname='Arial',
            fontsize='9',
            width='1.8',
            height='0.45',
            color='#333333',
            penwidth='1'
        )

    # 简化流程：5个主要模块
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Data Generation', style='rounded', color='#666666', bgcolor='#F5F5F5')
        add_compact_node('cfg', 'Load config\\n& materials', 'input')
        add_compact_node('tmm', 'TMM simulation\\n(with TTL cache)', 'physics')
        add_compact_node('val', 'Physical validation\\n|R+T+ε-1|<10⁻³', 'validation')
        c.edge('cfg', 'tmm')
        c.edge('tmm', 'val')

    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='ML Pipeline', style='rounded', color='#666666', bgcolor='#F5F5F5')
        add_compact_node('feat', 'Feature\\nengineering', 'ml')
        add_compact_node('split', 'Thickness-grouped\\nsplit', 'ml')
        add_compact_node('hpo', 'Halton + Bayesian\\noptimization', 'ml')
        add_compact_node('svr', 'SVR training\\n(RBF kernel)', 'ml')
        c.edge('val', 'feat')
        c.edge('feat', 'split')
        c.edge('split', 'hpo')
        c.edge('hpo', 'svr')

    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Evaluation', style='rounded', color='#666666', bgcolor='#F5F5F5')
        add_compact_node('eval', 'R², RMSE,\\nextrapolation', 'validation')
        add_compact_node('multi', 'Multi-region\\n(optional)', 'ml')
        add_compact_node('out', 'Emissivity\\nε(λ,d)', 'output')
        c.edge('svr', 'eval')
        c.edge('eval', 'multi', style='dashed')
        c.edge('multi', 'out', style='dashed')
        c.edge('eval', 'out')

    # 椭偏验证
    add_compact_node('ellip', 'Ellipsometry\\nvalidation', 'validation')
    dot.edge('out', 'ellip', style='dashed', color='#666666')

    output_path = os.path.join(output_dir, f'tmm_svr_simplified.{format}')
    dot.render(output_path, cleanup=True)
    print(f"简化版流程图已保存: {output_path}")

    return dot


def create_phase_detail(phase_name, output_dir='./output'):
    """
    创建特定Phase的详细子图（用于补充材料）
    """

    phases = {
        'phase2_tmm': create_phase2_detail,
        'phase6_hpo': create_phase6_detail,
        'phase9_multi': create_phase9_detail,
    }

    if phase_name in phases:
        return phases[phase_name](output_dir)
    else:
        raise ValueError(f"Unknown phase: {phase_name}")


def create_phase2_detail(output_dir):
    """Phase 2 TMM计算的详细流程"""

    dot = Digraph(
        name='Phase2_TMM_Detail',
        format='pdf',
        engine='dot'
    )

    dot.attr(rankdir='TB', bgcolor='white', fontname='Arial')

    # 输入参数
    dot.node('input', 'Input Parameters\\nλ, d_PDMS, d_SiO₂, (n,k) database',
             shape='box', style='filled', fillcolor='#E8F5E9')

    # 核心计算
    with dot.subgraph(name='cluster_loop') as c:
        c.attr(label='Parallel over wavelength-thickness grid', style='dashed')

        c.node('cache', 'Query TTL Cache\\nkey = hash(λ, d, n, k)',
               shape='diamond', style='filled', fillcolor='#E0F7FA')
        c.node('return_cache', 'Return cached\\nspectrum',
               shape='box', style='filled', fillcolor='#E0F7FA')

        # TMM核心
        c.node('delta', 'δⱼ = 2π(nⱼ\'+iκⱼ)dⱼ/λ',
               shape='box', style='filled', fillcolor='#E3F2FD')
        c.node('M_j', 'Mⱼ = [[cosδⱼ, i·sinδⱼ/ηⱼ],\\n[i·ηⱼ·sinδⱼ, cosδⱼ]]',
               shape='box', style='filled', fillcolor='#E3F2FD')
        c.node('M_sys', 'M = M_N·M_{N-1}·...·M₁\\n(ordered multiplication)',
               shape='box', style='filled', fillcolor='#E3F2FD', penwidth='2')
        c.node('RT', 'r = (η₀M₁₁ + η₀ηₛM₁₂ - M₂₁ - ηₛM₂₂) / (...)\\nR = |r|², T = |t|²',
               shape='box', style='filled', fillcolor='#E3F2FD')
        c.node('epsilon', 'ε = 1 - R - T',
               shape='box', style='filled', fillcolor='#E3F2FD')
        c.node('store', 'Store in cache\\nTTL = 180s',
               shape='box', style='filled', fillcolor='#E0F7FA')

        c.edge('cache', 'return_cache', label='Hit')
        c.edge('cache', 'delta', label='Miss')
        c.edge('delta', 'M_j')
        c.edge('M_j', 'M_sys')
        c.edge('M_sys', 'RT')
        c.edge('RT', 'epsilon')
        c.edge('epsilon', 'store')

    dot.edge('input', 'cache')

    output_path = os.path.join(output_dir, 'phase2_tmm_detail.pdf')
    dot.render(output_path, cleanup=True)
    print(f"Phase 2详细图已保存: {output_path}")
    return dot


def create_phase6_detail(output_dir):
    """Phase 6 超参优化的详细流程"""

    dot = Digraph(
        name='Phase6_HPO_Detail',
        format='pdf',
        engine='dot'
    )

    dot.attr(rankdir='TB', bgcolor='white', fontname='Arial')

    # 搜索空间
    dot.node('space', 'Hyperparameter Space\\nC ∈ [0.1, 1000], γ ∈ [0.001, 10], ε ∈ [0.001, 0.1]',
             shape='box', style='filled', fillcolor='#E8F5E9')

    # 三种搜索策略
    with dot.subgraph(name='cluster_methods') as c:
        c.attr(label='Search Strategies')

        c.node('grid', 'GridSearchCV\\n(exhaustive, slow)',
               shape='box', style='filled', fillcolor='#FFF3E0')
        c.node('random', 'RandomizedSearchCV\\n(uniform random)',
               shape='box', style='filled', fillcolor='#FFF3E0')
        c.node('halton', 'Halton Sequence\\n(quasi-random, low-discrepancy)\\nbases = (2,3) for (C,γ)',
               shape='box', style='filled', fillcolor='#FFF3E0', penwidth='2')
        c.node('bayesian', 'Bayesian Opt. (Optuna)\\nTPE sampler, adaptive',
               shape='box', style='filled', fillcolor='#FFF3E0')

    # 评估
    dot.node('cv', '5-Fold CV\\n(thickness-grouped)',
             shape='box', style='filled', fillcolor='#FFEBEE')
    dot.node('select', 'Select Best\\n(C*, γ*, ε*)',
             shape='box', style='filled', fillcolor='#FFFDE7', penwidth='2')

    dot.edge('space', 'grid')
    dot.edge('space', 'random')
    dot.edge('space', 'halton')
    dot.edge('halton', 'bayesian', label='refine', style='dashed')
    dot.edge('grid', 'cv', style='invis')
    dot.edge('random', 'cv', style='invis')
    dot.edge('bayesian', 'cv')
    c.edge('cv', 'select')

    output_path = os.path.join(output_dir, 'phase6_hpo_detail.pdf')
    dot.render(output_path, cleanup=True)
    print(f"Phase 6详细图已保存: {output_path}")
    return dot


def create_phase9_detail(output_dir):
    """Phase 9 多区域策略的详细流程"""

    dot = Digraph(
        name='Phase9_MultiRegion_Detail',
        format='pdf',
        engine='dot'
    )

    dot.attr(rankdir='TB', bgcolor='white', fontname='Arial')

    # 触发条件
    dot.node('trigger', 'Trigger: Single-region R² < 0.98\\nor λ-range > 5 μm',
             shape='diamond', style='filled', fillcolor='#F3E5F5')

    # 区域划分
    with dot.subgraph(name='cluster_split') as c:
        c.attr(label='Region Partitioning')

        c.node('split', 'RegionSplitter\\nλ-based division',
               shape='box', style='filled', fillcolor='#FFF3E0')
        c.node('regions', 'Regions: [2,5], [5,8], [8,11], [11,14] μm\\n(atmospheric window prioritized)',
               shape='box', style='filled', fillcolor='#FFF3E0')

    # 重叠区域
    with dot.subgraph(name='cluster_overlap') as c:
        c.attr(label='Overlap Handling (5%)', style='dashed')

        c.node('overlap_def', 'Ωₖ^overlap = [λ_{k+1}^min, λₖ^max]\\n|Ωₖ ∩ Ω_{k+1}| = 0.05 × min(|Ωₖ|, |Ω_{k+1}|)',
               shape='box', style='filled', fillcolor='#E8F5E9')
        c.node('weight', 'Weight calculation\\nwₖ = d_{k+1} / (dₖ + d_{k+1})\\ndₖ = |λ - λₖ^center| / |Ωₖ|',
               shape='box', style='filled', fillcolor='#E8F5E9')
        c.node('blend', 'Weighted blending\\nŷ = wₖ·ŷₖ + w_{k+1}·ŷ_{k+1}',
               shape='box', style='filled', fillcolor='#E8F5E9')

    # 路由
    dot.node('router',
             'MultiRegionModelRouter\\nPhase 1: Region extension (5%)\\nPhase 2: Weighted blending\\nPhase 3: Nearest-neighbor fallback',
             shape='box', style='filled', fillcolor='#FFFDE7', penwidth='2')

    dot.edge('trigger', 'split', label='Yes')
    dot.edge('split', 'regions')
    dot.edge('regions', 'overlap_def')
    dot.edge('overlap_def', 'weight')
    dot.edge('weight', 'blend')
    dot.edge('blend', 'router')

    output_path = os.path.join(output_dir, 'phase9_multiregion_detail.pdf')
    dot.render(output_path, cleanup=True)
    print(f"Phase 9详细图已保存: {output_path}")
    return dot


# ==================== 主程序 ====================

if __name__ == '__main__':

    # 创建输出目录
    output_dir = './workflow_figures'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("TMM-SVR Workflow Diagram Generator")
    print("=" * 60)

    # 1. 主流程图（完整版）
    print("\n[1/5] 生成完整版流程图...")
    create_tmm_svr_workflow(output_dir, format='pdf')
    create_tmm_svr_workflow(output_dir, format='png')
    create_tmm_svr_workflow(output_dir, format='svg')

    # 2. 简化版流程图（正文用）
    print("\n[2/5] 生成简化版流程图...")
    create_simplified_workflow(output_dir, format='pdf')
    create_simplified_workflow(output_dir, format='png')

    # 3. Phase详细子图
    print("\n[3/5] 生成Phase 2详细图...")
    create_phase2_detail(output_dir)

    print("\n[4/5] 生成Phase 6详细图...")
    create_phase6_detail(output_dir)

    print("\n[5/5] 生成Phase 9详细图...")
    create_phase9_detail(output_dir)

    print("\n" + "=" * 60)
    print(f"所有流程图已保存至: {os.path.abspath(output_dir)}")
    print("=" * 60)
    print("\n文件清单:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")
