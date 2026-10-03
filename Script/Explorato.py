# DEFINIÇÃO DAS BIBLIOTECAS PYTHON
import os
import time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

# Configurações de exibição e ambiente
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", context="notebook")

# Parâmetros e sementes globais
# Caminhos relativos à raiz do projeto (a pasta acima de "Script")
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:  # caso rode em Jupyter/notebook
    BASE_DIR = Path.cwd().parent

DATA_DIR = Path(os.environ.get("OLIST_DIR", BASE_DIR / "Dataset"))
FIG_DIR = BASE_DIR / "Resultados"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

def savefig(name):
    """Auxiliar para salvar e exibir gráficos gerados na EDA."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=130, bbox_inches="tight")
    plt.show()


# ANÁLISE EXPLORATÓRIA DA BASE DE DADOS (EDA)


def read_data(name, **kw):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"), **kw)

date_cols = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
             "order_delivered_customer_date", "order_estimated_delivery_date"]

orders    = read_data("olist_orders_dataset", parse_dates=date_cols)
customers = read_data("olist_customers_dataset")
items     = read_data("olist_order_items_dataset", parse_dates=["shipping_limit_date"])
reviews   = read_data("olist_order_reviews_dataset")
products  = read_data("olist_products_dataset")
trans     = read_data("product_category_name_translation")

# Análise de recorrência por cliente único
ped_cli = orders.merge(customers, on="customer_id")
pedidos_por_cliente = ped_cli.groupby("customer_unique_id").order_id.nunique()

entregues = (items.merge(orders[["order_id", "customer_id", "order_status"]], on="order_id")
                  .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id"))
entregues = entregues[entregues.order_status == "delivered"]
pares = entregues.drop_duplicates(["customer_unique_id", "product_id"])

n_u, n_i = pares.customer_unique_id.nunique(), pares.product_id.nunique()
esparsidade = 1 - len(pares) / (n_u * n_i)
compradores_por_prod = pares.groupby("product_id").customer_unique_id.nunique()
prod_por_user = pares.groupby("customer_unique_id").product_id.nunique()

print(f"   * Usuários Únicos: {n_u:,}")
print(f"   * Produtos Únicos: {n_i:,}")
print(f"   * Interações Válidas: {len(pares):,}")
print(f"   * Esparsidade da Matriz: {esparsidade * 100:.4f}%")
print(f"   * % Clientes com 2+ pedidos: {(pedidos_por_cliente >= 2).mean() * 100:.2f}%")

# Plotagem dos Gráficos Diagnósticos da EDA
prod_en = products.merge(trans, on="product_category_name", how="left")
prod_en["categoria"] = prod_en.product_category_name_english.fillna(prod_en.product_category_name).fillna("unknown")
cat_vendas = (items.merge(prod_en[["product_id", "categoria"]], on="product_id")
                   .groupby("categoria").agg(itens=("order_id", "size")))

# Cada gráfico é gerado e salvo separadamente em Resultados/

# 1) Volume mensal
m = orders.set_index("order_purchase_timestamp").resample("MS").size()
m = m[(m.index >= "2017-01-01") & (m.index <= "2018-08-01")]
plt.figure(figsize=(8, 5))
plt.plot(m.index, m.values, marker="o", color="#1f77b4")
plt.title("Pedidos por mês (jan/2017–ago/2018)")
plt.xlabel("Mês")
plt.ylabel("Pedidos")
savefig("eda_01_pedidos_por_mes.png")

# 2) Pedidos por cliente
ped_cli_cont = pedidos_por_cliente.clip(upper=5).value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(ped_cli_cont.index.astype(str), ped_cli_cont.values, color="#4c72b0")
plt.yscale("log")
plt.title("Pedidos por cliente (5 = 5+), escala log")
plt.xlabel("Nº de pedidos")
plt.ylabel("Clientes (log)")
savefig("eda_02_pedidos_por_cliente.png")

# 3) Cauda longa
plt.figure(figsize=(8, 5))
plt.hist(compradores_por_prod.clip(upper=30), bins=30, color="#55a868")
plt.yscale("log")
plt.title("Cauda longa: compradores por produto")
plt.xlabel("Compradores por produto (30 = 30+)")
plt.ylabel("Produtos (log)")
savefig("eda_03_cauda_longa.png")

# 4) Top categorias
top = cat_vendas.sort_values("itens").tail(12)
plt.figure(figsize=(8, 6))
plt.barh(top.index, top.itens, color="#c44e52")
plt.title("Top 12 categorias (itens vendidos)")
plt.xlabel("Itens vendidos")
savefig("eda_04_top_categorias.png")

# 5) Distribuição das notas
notas = reviews.review_score.value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(notas.index, notas.values, color="#8172b2")
plt.title("Distribuição das notas")
plt.xlabel("Nota")
plt.ylabel("Avaliações")
savefig("eda_05_distribuicao_notas.png")

# 6) Preços
plt.figure(figsize=(8, 5))
plt.hist(np.log10(items.price), bins=50, color="#ccb974")
plt.title("Preço dos itens (log10 R$)")
plt.xlabel("log10(preço em R$)")
plt.ylabel("Itens")
savefig("eda_06_precos.png")


#  TRATAMENTO E PREPARAÇÃO A BASE DE DADOS PARA O TREINAMENTO



rv_pedido = reviews.groupby("order_id").review_score.mean().rename("nota")

inter = (items[["order_id", "order_item_id", "product_id"]]
         .merge(orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp"]], on="order_id")
         .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id"))

# Regra de negócio: Considerar apenas transações concluídas e bem avaliadas
inter = inter[inter.order_status == "delivered"]
inter["nota"] = inter.order_id.map(rv_pedido)
inter = inter[~(inter.nota <= 2)]

inter = inter.sort_values(["order_purchase_timestamp", "order_id", "order_item_id"])
ui = (inter.groupby(["customer_unique_id", "product_id"], sort=False)
           .agg(ts=("order_purchase_timestamp", "first"), order_id=("order_id", "first"), pos=("order_item_id", "first"))
           .reset_index()
           .sort_values(["customer_unique_id", "ts", "order_id", "pos", "product_id"])
           .reset_index(drop=True))

# Criação dos mapeamentos e dicionários de índices
catalog = products.product_id.values
item2idx = pd.Series(np.arange(len(catalog)), index=catalog)
user_ids = ui.customer_unique_id.unique()
user2idx = pd.Series(np.arange(len(user_ids)), index=user_ids)

ui["u"] = ui.customer_unique_id.map(user2idx).values
ui["i"] = ui.product_id.map(item2idx).values
N_USERS, N_ITEMS = len(user_ids), len(catalog)

# Divisão Temporal Leave-One-Out (A última compra dos usuários elegíveis vai para validação/teste)
n_prod_user = ui.groupby("u").i.transform("size")
rank_desc = ui.groupby("u").cumcount(ascending=False)
mask_held = (n_prod_user >= 2) & (rank_desc == 0)

held = ui[mask_held].copy()
train = ui[~mask_held].copy()

# Separação do grupo retido entre Validação (30%) e Teste (70%)
elig = held.u.values
perm = rng.permutation(len(elig))
n_val = int(0.3 * len(elig))

val_idx, test_idx = perm[:n_val], perm[n_val:]
val_u, val_t = held.u.values[val_idx], held.i.values[val_idx]
test_u, test_t = held.u.values[test_idx], held.i.values[test_idx]

# Matriz Esparsa de Treino (Usuário-Item)
R_train = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (train.u.values, train.i.values)), shape=(N_USERS, N_ITEMS))
item_cnt = np.asarray(R_train.sum(0)).ravel()
pop_n = np.log1p(item_cnt) / np.log1p(item_cnt).max()

print(f"   * Matriz Usuário-Item criada: {R_train.shape[0]} usuários x {R_train.shape[1]} produtos.")
print(f"   * Usuários para Validação: {len(val_u)} | Usuários para Teste: {len(test_u)}")

# DEFINIR A TÉCNICA E REALIZAR O TREINAMENTO (PoC)



def build_item_knn(R, shrink=20):
    """
    Treinamento do Modelo: Calcula a matriz de similaridade de cosseno amortecida.
    """
    deg = np.asarray(R.sum(1)).ravel()
    Rm = R[deg >= 2]
    C = (Rm.T @ Rm).tocoo()
    keep = C.row != C.col
    r, c, d = C.row[keep], C.col[keep], C.data[keep]
    n = np.maximum(np.asarray(R.sum(0)).ravel(), 1)
    
    # Cosine Similarity com Shrinkage
    similarity_matrix = sp.csr_matrix(
        (d / (np.sqrt(n[r] * n[c]) + shrink), (r, c)), 
        shape=(R.shape[1], R.shape[1])
    )
    return similarity_matrix

# Treinamento da matriz de similaridade
start_train = time.time()
S_knn = build_item_knn(R_train, shrink=20)
print(f"   * Treinamento concluído em {time.time() - start_train:.2f} segundos!")

def predict_item_knn(users):
    """Gera os scores de recomendação para uma lista de usuários."""
    return (R_train[users] @ S_knn).toarray()



# AVALIAÇÃO DE DESEMPENHO DO MODELO


print("\nAvaliando o Desempenho do Modelo no Conjunto de Teste...")

BATCH = 400
EPS = 1e-9

def mask_seen(S, H):
    """Mascara produtos que o usuário já comprou no treino."""
    c = H.tocoo()
    S[c.row, c.col] = -1e9
    return S

def rank_of_target(S, targets):
    """Calcula a posição (rank) do produto alvo real no ranking gerado."""
    tv = S[np.arange(len(targets)), targets]
    return (S >= tv[:, None]).sum(1)

def evaluate_poc(predict_fn, users, targets, ks=(5, 10, 20)):
    """Mecanismo de avaliação do modelo."""
    ranks = []
    for s in range(0, len(users), BATCH):
        u, t = users[s:s + BATCH], targets[s:s + BATCH]
        
        # Predição + Desempate por popularidade
        S = np.asarray(predict_fn(u), dtype=np.float64) + EPS * pop_n
        S = mask_seen(S, R_train[u])
        ranks.append(rank_of_target(S, t))
        
    ranks = np.concatenate(ranks)
    
    # Cálculo das métricas
    metrics = {}
    for k in ks:
        metrics[f"HR@{k}"] = float((ranks <= k).mean())
        metrics[f"NDCG@{k}"] = float(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0).mean())
    metrics["MRR@20"] = float(np.where(ranks <= 20, 1 / ranks, 0).mean())
    
    return metrics

# Execução da Avaliação Final
results = evaluate_poc(predict_item_knn, test_u, test_t)

print("\n===============================================================================")
print("RESULTADOS FINAIS DE DESEMPENHO (MÉTRICAS NO TESTE)")
print("===============================================================================")
print(f" Hit Rate @ 5   (HR@5)   : {results['HR@5'] * 100:.2f}%")
print(f" Hit Rate @ 10  (HR@10)  : {results['HR@10'] * 100:.2f}%")
print(f" Hit Rate @ 20  (HR@20)  : {results['HR@20'] * 100:.2f}%")
print(f" NDCG @ 10      (NDCG@10): {results['NDCG@10']:.4f}")
print(f" MRR @ 20       (MRR@20) : {results['MRR@20']:.4f}")
print("===============================================================================")