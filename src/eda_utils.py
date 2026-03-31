import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap

# 1. CONFIGURAÇÃO DE IDENTIDADE VISUAL GLOBAL
INSTYLE_PALETTE = ['#8d0801', '#8a817c', '#461220', '#e39695', '#cc444b']
sns.set_theme(style='whitegrid', palette=sns.color_palette(INSTYLE_PALETTE))
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.titlesize': 16,
    'figure.titleweight': 'bold'
})

# Mapa de cores personalizado para Correlações (usando tons da paleta)
instyle_cmap = LinearSegmentedColormap.from_list("instyle", [INSTYLE_PALETTE[0], "#ffffff", INSTYLE_PALETTE[2]])

def analise_mensal_cpa(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    # Agregação
    monthly = df.groupby('month').agg({'spend': 'sum', 'conversions': 'sum'}).reset_index()
    monthly['monthly_cpa'] = monthly['spend'] / monthly['conversions']
    monthly['cpa_variation_pct'] = monthly['monthly_cpa'].pct_change() * 100
    monthly['month_str'] = monthly['month'].astype(str)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Barras: CPA Médio
    ax1.bar(monthly['month_str'], monthly['monthly_cpa'], color=INSTYLE_PALETTE[1], alpha=0.4, label='CPA Médio Mensal')
    ax1.set_ylabel('CPA (R$)', color=INSTYLE_PALETTE[1])
    
    # Linha: Variação %
    ax2 = ax1.twinx()
    ax2.plot(monthly['month_str'], monthly['cpa_variation_pct'], color=INSTYLE_PALETTE[0], marker='o', linewidth=3, label='Variação MoM (%)')
    ax2.set_ylabel('Variação Mensal (%)', color=INSTYLE_PALETTE[0])
    
    plt.title('Performance Mensal: Evolução e Variação Percentual do CPA', pad=20)
    ax1.set_xlabel('Mês')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    # Legendas combinadas
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()
    return monthly

def dossie_visual_marketing(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # BLOCO 1: BOXPLOTS
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [('cpc', 3), ('cpa', 4), ('roas', 0)]
    
    for i, (col, color_idx) in enumerate(metrics):
        sns.boxplot(y=df[col], ax=axes1[i], color=INSTYLE_PALETTE[color_idx])
        axes1[i].set_title(f'Distribuição: {col.upper()}')
    
    fig1.suptitle('Diagnóstico de Outliers e Amplitude (Bloco 1)')
    plt.tight_layout()
    plt.show()

    # BLOCO 2: SCATTERS
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    scatters = [
        ('cpc', 'cpa', 0, 'Inflação de Leilão: CPA vs CPC'),
        ('ctr', 'cpa', 2, 'Engajamento: CPA vs CTR'),
        ('frequency', 'cpa', 3, 'Fadiga de Criativo: CPA vs Frequência'),
        ('reach', 'cpa', 4, 'Escassez de Público: CPA vs Alcance')
    ]
    
    for idx, (x, y, color_idx, title) in enumerate(scatters):
        ax = axes2[idx // 2, idx % 2]
        sns.scatterplot(data=df, x=x, y=y, ax=ax, color=INSTYLE_PALETTE[color_idx], alpha=0.6)
        ax.set_title(title)
    
    fig2.suptitle('Matriz de Correlação Gráfica (Bloco 2)')
    plt.tight_layout()
    plt.show()

    # BLOCO 3: SÉRIES TEMPORAIS
    df['month'] = df['date'].dt.to_period('M')
    df['revenue'] = df['spend'] * df['roas']
    
    monthly = df.groupby('month').agg({
        'frequency': 'mean', 'reach': 'sum', 'cpc': 'mean', 
        'conversions': 'sum', 'spend': 'sum', 'revenue': 'sum'
    }).reset_index()
    
    monthly['weighted_roas'] = monthly['revenue'] / monthly['spend']
    monthly['month_str'] = monthly['month'].astype(str)
    
    metrics_ts = [
        ('frequency', 0, 'Frequência Média'),
        ('reach', 1, 'Alcance Total (Reach)'),
        ('cpc', 2, 'Custo Por Clique (CPC)'),
        ('conversions', 4, 'Conversões Totais'),
        ('weighted_roas', 3, 'ROAS Ponderado (Retorno)')
    ]
    
    fig3, axes3 = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
    for i, (col, color_idx, title) in enumerate(metrics_ts):
        axes3[i].plot(monthly['month_str'], monthly[col], color=INSTYLE_PALETTE[color_idx], marker='o', linewidth=3)
        axes3[i].set_title(f'Série Temporal: {title}')
        if col == 'weighted_roas':
            axes3[i].axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Break-even')
            axes3[i].legend()
            
    plt.xticks(rotation=45)
    fig3.suptitle('Dinâmica Operacional ao Longo dos Meses (Bloco 3)')
    plt.tight_layout()
    plt.show()

def analise_correlacao_avancada(file_path):
    df = pd.read_csv(file_path)
    num_cols = ['reach', 'frequency', 'clicks', 'ctr', 'conversions', 'spend', 'cpc', 'cpa', 'roas']
    df_num = df[num_cols]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    methods = [('pearson', 'Lineares Diretas'), ('spearman', 'Não-Lineares / Escala')]
    
    for i, (method, title) in enumerate(methods):
        sns.heatmap(df_num.corr(method=method), annot=True, fmt=".2f", 
                    cmap=instyle_cmap, vmax=1, vmin=-1, center=0, 
                    ax=axes[i], square=True, cbar_kws={"shrink": .8})
        axes[i].set_title(f'Correlação de {method.capitalize()} ({title})')
    
    plt.tight_layout()
    plt.show()


def analise_limites_eficiencia(file_path):
    """Analisa o ponto de saturação da frequência e os limites de leilão (CPC) em relação ao CPA."""
    df = pd.read_csv(file_path)
    
    # 1. Processamento da Frequência (Degraus e Marginal)
    df['freq_bin'] = np.floor(df['frequency'])
    tabela_freq = df.groupby('freq_bin')['cpa'].mean().reset_index()
    tabela_freq['delta_cpa'] = tabela_freq['cpa'].diff()

    # 2. Processamento do CPC (Decis)
    df['cpc_decil'] = pd.qcut(df['cpc'], q=10, duplicates='drop')
    tabela_cpc = df.groupby('cpc_decil')['cpa'].mean().reset_index()
    tabela_cpc['cpc_decil_str'] = tabela_cpc['cpc_decil'].astype(str)

    # 3. Plotagem das Provas Matemáticas
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Gráfico A: A Escada da Frequência (Barras em cinza/bege, Linha em vermelho escuro)
    sns.barplot(data=tabela_freq, x='freq_bin', y='cpa', color=INSTYLE_PALETTE[1], alpha=0.5, ax=axes[0])
    ax0_twin = axes[0].twinx()
    sns.lineplot(data=tabela_freq, x=tabela_freq.index, y='delta_cpa', color=INSTYLE_PALETTE[0], marker='o', linewidth=3, ax=ax0_twin)
    axes[0].set_title('Saturação de Frequência: CPA Médio vs Custo Marginal')
    axes[0].set_xlabel('Frequência de Exibição')
    ax0_twin.set_ylabel('Aceleração do Custo (Delta CPA)')

    # Gráfico B: O Limite do Leilão (Barras em vinho)
    sns.barplot(data=tabela_cpc, x='cpc_decil_str', y='cpa', color=INSTYLE_PALETTE[2], ax=axes[1])
    axes[1].set_title('Limite do Leilão: CPA Médio por Faixa de CPC')
    axes[1].set_xlabel('Faixas de Preço do CPC (Decis)')
    plt.sca(axes[1])
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    
    return tabela_freq


def analise_funil_conversao(file_path):
    """Gera o diagnóstico de fricção e volume do funil de vendas."""
    df = pd.read_csv(file_path)

    # 1. Filtro de Realidade: Garantindo a métrica de topo de funil
    if 'impressions' not in df.columns:
        df['impressions'] = df['clicks'] / df['ctr']

    # 2. Definindo a Jornada (ajustável conforme colunas do CSV)
    etapas_disponiveis = [col for col in ['impressions', 'clicks', 'conversions'] if col in df.columns]

    # 3. Processamento dos Dados
    funil_totais = df[etapas_disponiveis].sum().reset_index()
    funil_totais.columns = ['etapa', 'volume']

    # 4. Cálculo de Retenção e Drop-off
    funil_totais['taxa_retencao'] = funil_totais['volume'] / funil_totais['volume'].shift(1)
    funil_totais['taxa_retencao'] = funil_totais['taxa_retencao'].fillna(1.0)
    funil_totais['drop_off_rate'] = 1 - funil_totais['taxa_retencao']

    funil_totais['drop_off_str'] = (funil_totais['drop_off_rate'] * 100).round(2).astype(str) + '%'

    # 5. Visualização
    plt.figure(figsize=(12, 6))
    # Usando o vinho da paleta (index 2) para as barras
    ax = sns.barplot(x='volume', y='etapa', data=funil_totais, color=INSTYLE_PALETTE[2], orient='h')

    # Rótulos de Drop-off (usando o vermelho index 0 para destaque)
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if i > 0:
            drop_off = funil_totais['drop_off_str'].iloc[i]
            ax.text(width + (width * 0.02), p.get_y() + p.get_height()/2.,
                    f'🔻 Perda de {drop_off}',
                    ha="left", va="center", color=INSTYLE_PALETTE[0], weight='bold', fontsize=12)

    plt.title('Diagnóstico de Fricção: Volume Absoluto e Drop-off Rate', pad=20)
    plt.xlabel('Volume de Usuários (Absoluto)')
    plt.ylabel('Etapas da Jornada')
    plt.tight_layout()
    plt.show()

    return funil_totais


def analise_modelo_elasticidade(file_path):
    """Executa regressão Log-Log para entender a sensibilidade do CPA às métricas."""
    df = pd.read_csv(file_path)
    
    features = ['frequency', 'clicks', 'ctr', 'cpc']
    
    # 1. Limpeza e Transformação Logarítmica
    # Filtramos apenas valores > 0 (log de zero ou negativo não existe)
    df_log = df[features + ['cpa']].dropna()
    df_log = df_log[(df_log[features + ['cpa']] > 0).all(axis=1)]
    
    X_log = np.log(df_log[features])
    y_log = np.log(df_log['cpa'])
    X_log_const = sm.add_constant(X_log)

    # 2. Ajuste do Modelo OLS
    modelo = sm.OLS(y_log, X_log_const).fit()

    # 3. Print Executivo no Console
    print("\n" + "="*50)
    print("--- RADIOGRAFIA DE ELASTICIDADE (MODELO LOG-LOG) ---")
    print(f"R-squared: {modelo.rsquared:.4f}")
    print("Interpretando: Cada 1% de aumento na feature altera o CPA em X%")
    print("="*50)
    print(modelo.summary().tables[1]) # Mostra apenas a tabela de coeficientes principal

    # 4. Visualização dos Coeficientes (Elasticidade)
    plt.figure(figsize=(10, 6))
    coefs = modelo.params.drop('const')
    # Usando o tom de vinho da sua paleta (index 2)
    coefs.plot(kind='barh', color=INSTYLE_PALETTE[2])
    
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('Elasticidade: Impacto Percentual no CPA', pad=20)
    plt.xlabel('Coeficiente de Elasticidade (Impacto de 1%)')
    plt.ylabel('Métricas')
    plt.tight_layout()
    plt.show()

    return modelo



def analise_sazonalidade_semanal(file_path, coluna_data='date'):
    """Analisa a eficiência do CPA e volume de conversões por dia da semana."""
    df = pd.read_csv(file_path)

    if coluna_data not in df.columns:
        print(f"⚠️ Erro: Coluna '{coluna_data}' não encontrada no CSV.")
        return None

    # 1. Tratamento de Datas e Mapeamento
    df['data_formatada'] = pd.to_datetime(df[coluna_data])
    df['dia_semana'] = df['data_formatada'].dt.dayofweek
    mapa_dias = {0: '1. Seg', 1: '2. Ter', 2: '3. Qua', 3: '4. Qui', 4: '5. Sex', 5: '6. Sáb', 6: '7. Dom'}
    df['nome_dia'] = df['dia_semana'].map(mapa_dias)

    # 2. Agregação
    sazonalidade = df.groupby('nome_dia').agg({
        'cpa': 'mean',
        'conversions': 'sum'
    }).reset_index()

    # 3. Visualização
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Barras: Volume de Conversões (Vinho da paleta)
    sns.barplot(data=sazonalidade, x='nome_dia', y='conversions', color=INSTYLE_PALETTE[2], alpha=0.6, ax=ax1)
    ax1.set_ylabel('Volume de Conversões', color=INSTYLE_PALETTE[2])

    # Linha: CPA Médio (Vermelho de alerta)
    ax2 = ax1.twinx()
    sns.lineplot(data=sazonalidade, x='nome_dia', y='cpa', color=INSTYLE_PALETTE[0], marker='o', linewidth=3, ax=ax2)
    ax2.set_ylabel('CPA Médio (R$)', color=INSTYLE_PALETTE[0])

    plt.title('Micro-Sazonalidade: Eficiência vs Volume por Dia da Semana', pad=20)
    plt.tight_layout()
    plt.show()

    return sazonalidade


def analise_modelo_hibrido(file_path, coluna_data='date'):
    """Modelo definitivo: Une métricas de leilão e sazonalidade para explicar o CPA."""
    df = pd.read_csv(file_path)
    
    # 1. Preparação da Sazonalidade (Calendário)
    if coluna_data in df.columns:
        df['data_formatada'] = pd.to_datetime(df[coluna_data])
        df['dia_semana'] = df['data_formatada'].dt.dayofweek
        mapa_dias = {0: '1. Seg', 1: '2. Ter', 2: '3. Qua', 3: '4. Qui', 4: '5. Sex', 5: '6. Sáb', 6: '7. Dom'}
        df['nome_dia'] = df['dia_semana'].map(mapa_dias)
    else:
        print(f"⚠️ Erro: Coluna '{coluna_data}' não encontrada.")
        return None

    # 2. Limpeza e Filtro Logarítmico
    features_leilao = ['frequency', 'cpc']
    df_hibrido = df[features_leilao + ['cpa', 'nome_dia']].dropna()
    df_hibrido = df_hibrido[(df_hibrido[features_leilao + ['cpa']] > 0).all(axis=1)]

    # 3. Transformações (Log + Dummies)
    X_leilao_log = np.log(df_hibrido[features_leilao])
    y_log = np.log(df_hibrido['cpa'])
    X_sazonal = pd.get_dummies(df_hibrido['nome_dia'], drop_first=True).astype(int)
    
    X_final = pd.concat([X_leilao_log, X_sazonal], axis=1)
    X_final_const = sm.add_constant(X_final)

    # 4. Ajuste do Modelo OLS
    modelo = sm.OLS(y_log, X_final_const).fit()

    # 5. Gráfico de Impacto Relativo
    plt.figure(figsize=(12, 6))
    coefs = modelo.params.drop('const').sort_values()
    
    # Cores: Vinho (index 2) para o leilão e Cinza (index 1) para os dias
    cores_plot = [INSTYLE_PALETTE[2] if c in features_leilao else INSTYLE_PALETTE[1] for c in coefs.index]
    
    coefs.plot(kind='barh', color=cores_plot)
    plt.axvline(0, color=INSTYLE_PALETTE[0], linestyle='--', alpha=0.5)
    plt.title('Impacto no CPA: Leilão (Vinho) vs. Sazonalidade (Cinza)')
    plt.xlabel('Coeficiente (Elasticidade / Peso do Dia)')
    plt.tight_layout()
    plt.show()

    return modelo


def analise_vintage_semanal(file_path, coluna_data='date'):
    """Analisa a degradação da performance por safras semanais."""
    df = pd.read_csv(file_path)

    if coluna_data not in df.columns:
        print(f"⚠️ Erro: Coluna '{coluna_data}' não encontrada.")
        return None

    # 1. Tratamento de Tempo e Agrupamento por Safra (Week)
    df['data_formatada'] = pd.to_datetime(df[coluna_data])
    df['safra_semana'] = df['data_formatada'].dt.to_period('W')

    # 2. Agregação das Métricas
    vintage = df.groupby('safra_semana').agg({
        'cpa': 'mean',
        'conversions': 'sum',
        'spend': 'sum'
    }).reset_index()

    vintage['safra_str'] = vintage['safra_semana'].astype(str)

    # 3. Visualização do Desgaste
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Barras: Volume de Conversões (Vinho)
    sns.barplot(data=vintage, x='safra_str', y='conversions', color=INSTYLE_PALETTE[2], alpha=0.7, ax=ax1)
    ax1.set_ylabel('Volume de Conversões (Safra)', color=INSTYLE_PALETTE[2])
    ax1.set_xlabel('Semana da Campanha')
    plt.xticks(rotation=45)

    # Linha: Degradação do CPA (Vermelho)
    ax2 = ax1.twinx()
    sns.lineplot(data=vintage, x='safra_str', y='cpa', color=INSTYLE_PALETTE[0], marker='o', linewidth=3, ax=ax2)
    ax2.set_ylabel('CPA Médio (R$)', color=INSTYLE_PALETTE[0])

    plt.title('Desgaste do Público: Volume vs. Degradação do CPA por Safra', pad=20)
    plt.tight_layout()
    plt.show()

    return vintage


def analise_modelo_exaustao(file_path, coluna_data='date'):
    """Mede o impacto do envelhecimento da campanha (tempo) vs. métricas de leilão."""
    df = pd.read_csv(file_path)
    
    # 1. Criação do Índice de Envelhecimento (Time Trend)
    df['data_formatada'] = pd.to_datetime(df[coluna_data])
    data_minima = df['data_formatada'].min()
    df['week_index'] = ((df['data_formatada'] - data_minima).dt.days // 7) + 1

    # 2. Limpeza e Log-Transform
    features_leilao = ['frequency', 'cpc']
    df_ex = df[features_leilao + ['cpa', 'week_index']].dropna()
    df_ex = df_ex[(df_ex[features_leilao + ['cpa']] > 0).all(axis=1)]

    X_leilao_log = np.log(df_ex[features_leilao])
    y_log = np.log(df_ex['cpa'])
    X_time = df_ex['week_index']

    # 3. Modelo Híbrido: Leilão + Tempo
    X_final = pd.concat([X_leilao_log, X_time], axis=1)
    X_final_const = sm.add_constant(X_final)
    modelo = sm.OLS(y_log, X_final_const).fit()

    # 4. Visualização de Impacto (Coeficientes)
    plt.figure(figsize=(10, 6))
    coefs = modelo.params.drop('const').sort_values()
    
    # Cores: Tons de vinho e cinza da paleta
    cores = [INSTYLE_PALETTE[0] if c == 'week_index' else INSTYLE_PALETTE[2] for c in coefs.index]
    
    coefs.plot(kind='barh', color=cores)
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    plt.title('Diagnóstico de Exaustão: Leilão (Vinho) vs. Tempo (Vermelho)')
    plt.xlabel('Impacto no CPA (Coeficiente)')
    plt.tight_layout()
    plt.show()

    return modelo

def analise_exaustao_preditiva(file_path, coluna_data='date'):
    """Gera previsões de CPA baseadas no envelhecimento e plota a curva de exaustão."""
    df = pd.read_csv(file_path)
    
    # 1. Preparação e Índice de Tempo
    df['data_formatada'] = pd.to_datetime(df[coluna_data])
    data_minima = df['data_formatada'].min()
    df['week_index'] = ((df['data_formatada'] - data_minima).dt.days // 7) + 1

    features_leilao = ['frequency', 'cpc']
    df_ex = df[features_leilao + ['cpa', 'week_index']].dropna()
    df_ex = df_ex[(df_ex[features_leilao + ['cpa']] > 0).all(axis=1)]

    # 2. Modelo Log-Log (Semi-log para o tempo)
    X_leilao_log = np.log(df_ex[features_leilao])
    y_log = np.log(df_ex['cpa'])
    X_time = df_ex['week_index']
    X_final = pd.concat([X_leilao_log, X_time], axis=1)
    X_final_const = sm.add_constant(X_final)
    
    modelo = sm.OLS(y_log, X_final_const).fit()

    # 3. Gerando as Previsões (Voltando do Log para Reais)
    df_ex['cpa_previsto'] = np.exp(modelo.predict(X_final_const))

    # 4. Diagnóstico Visual
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Gráfico A: Real vs Previsto
    sns.scatterplot(data=df_ex, x='cpa', y='cpa_previsto', alpha=0.5, color=INSTYLE_PALETTE[1], ax=axes[0])
    max_val = max(df_ex['cpa'].max(), df_ex['cpa_previsto'].max())
    axes[0].plot([0, max_val], [0, max_val], color=INSTYLE_PALETTE[0], linestyle='--', linewidth=2)
    axes[0].set_title(f'Aderência do Modelo (R² = {modelo.rsquared:.3f})')
    axes[0].set_xlabel('CPA Real (R$)')
    axes[0].set_ylabel('CPA Previsto (R$)')

    # Gráfico B: Curva de Exaustão (Tendência)
    avg_trend = df_ex.groupby('week_index')['cpa_previsto'].mean().reset_index()
    sns.lineplot(data=avg_trend, x='week_index', y='cpa_previsto', color=INSTYLE_PALETTE[0], linewidth=4, marker='o', markersize=10, ax=axes[1])
    axes[1].set_title('A Curva de Exaustão: Encarecimento por Semana')
    axes[1].set_xlabel('Semanas de Campanha')
    axes[1].set_ylabel('CPA Médio Estimado (R$)')

    plt.tight_layout()
    plt.show()

    # Cálculo do coeficiente de impacto semanal
    impacto_semanal = (np.exp(modelo.params['week_index']) - 1) * 100
    print(f"--- DIAGNÓSTICO ESTRATÉGICO ---")
    print(f"Impacto do Tempo: A cada semana, o CPA base sobe aproximadamente {impacto_semanal:.2f}%")
    
    return modelo, df_ex