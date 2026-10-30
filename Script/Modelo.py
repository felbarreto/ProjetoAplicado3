# ETAPA 3 — AJUSTE E REAVALIAÇÃO DO MODELO
# Mesmo tratamento de dados e mesma divisão do Explorato.py (Etapa 2), para que a
# comparação entre a PoC e o modelo ajustado seja justa.
import os
import time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

DATA_DIR = Path(os.environ.get("OLIST_DIR", BASE_DIR / "Dataset"))
OUT_DIR = BASE_DIR / "Resultados"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)


# CARGA E PREPARAÇÃO DOS DADOS (idêntico à Etapa 2)


def read_data(name, **kw):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"), **kw)

orders    = read_data("olist_orders_dataset", parse_dates=["order_purchase_timestamp"])
customers = read_data("olist_customers_dataset")
items     = read_data("olist_order_items_dataset")
reviews   = read_data("olist_order_reviews_dataset")
products  = read_data("olist_products_dataset")

rv_pedido = reviews.groupby("order_id").review_score.mean().rename("nota")

inter = (items[["order_id", "order_item_id", "product_id"]]
         .merge(orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp"]], on="order_id")
         .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id"))

# Regra de negócio: apenas pedidos entregues e sem avaliação negativa (nota 1 ou 2)
inter = inter[inter.order_status == "delivered"]
inter["nota"] = inter.order_id.map(rv_pedido)
inter = inter[~(inter.nota <= 2)]

inter = inter.sort_values(["order_purchase_timestamp", "order_id", "order_item_id"])
ui = (inter.groupby(["customer_unique_id", "product_id"], sort=False)
           .agg(ts=("order_purchase_timestamp", "first"), order_id=("order_id", "first"), pos=("order_item_id", "first"))
           .reset_index()
           .sort_values(["customer_unique_id", "ts", "order_id", "pos", "product_id"])
           .reset_index(drop=True))

catalog = products.product_id.values
item2idx = pd.Series(np.arange(len(catalog)), index=catalog)
user_ids = ui.customer_unique_id.unique()
user2idx = pd.Series(np.arange(len(user_ids)), index=user_ids)

ui["u"] = ui.customer_unique_id.map(user2idx).values
ui["i"] = ui.product_id.map(item2idx).values
N_USERS, N_ITEMS = len(user_ids), len(catalog)

# Leave-one-out temporal: o último produto de quem tem 2+ produtos é o alvo
n_prod_user = ui.groupby("u").i.transform("size")
rank_desc = ui.groupby("u").cumcount(ascending=False)
mask_held = (n_prod_user >= 2) & (rank_desc == 0)

held = ui[mask_held].copy()
train = ui[~mask_held].copy()

perm = rng.permutation(len(held))
n_val = int(0.3 * len(held))
val_idx, test_idx = perm[:n_val], perm[n_val:]
val_u, val_t = held.u.values[val_idx], held.i.values[val_idx]
test_u, test_t = held.u.values[test_idx], held.i.values[test_idx]

R_train = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (train.u.values, train.i.values)), shape=(N_USERS, N_ITEMS))
item_cnt = np.asarray(R_train.sum(0)).ravel()
pop_n = np.log1p(item_cnt) / np.log1p(item_cnt).max()

print(f"   * Matriz Usuário-Item: {N_USERS} usuários x {N_ITEMS} produtos")
print(f"   * Usuários para Validação: {len(val_u)} | Usuários para Teste: {len(test_u)}")


# COMPONENTES DO MODELO


def build_item_knn(R, shrink=20):
    """Similaridade de cosseno item-item com shrinkage (mesma técnica da PoC)."""
    deg = np.asarray(R.sum(1)).ravel()
    Rm = R[deg >= 2]
    C = (Rm.T @ Rm).tocoo()
    keep = C.row != C.col
    r, c, d = C.row[keep], C.col[keep], C.data[keep]
    n = np.maximum(np.asarray(R.sum(0)).ravel(), 1)
    return sp.csr_matrix((d / (np.sqrt(n[r] * n[c]) + shrink), (r, c)), shape=(R.shape[1], R.shape[1]))

# Conteúdo: categoria do produto. O perfil do usuário é a fração de compras por categoria.
cat_codes = products.product_category_name.fillna("unknown").astype("category").cat.codes.values
N_CATS = cat_codes.max() + 1
I2C = sp.csr_matrix((np.ones(N_ITEMS, dtype=np.float32), (np.arange(N_ITEMS), cat_codes)), shape=(N_ITEMS, N_CATS))
U2C = R_train @ I2C
U2C = sp.csr_matrix(U2C.multiply(1 / np.maximum(U2C.sum(1), 1)))

def content_scores(users):
    return (U2C[users] @ I2C.T).toarray()

def knn_scores(users, S):
    """Scores do Item-kNN normalizados para [0, 1] por usuário."""
    X = (R_train[users] @ S).toarray()
    return X / np.maximum(X.max(1, keepdims=True), 1e-9)

def make_hybrid(S, w_knn, w_cont, w_pop):
    """Híbrido ponderado: soma dos componentes normalizados, cada um com seu peso."""
    def predict(users):
        return w_knn * knn_scores(users, S) + w_cont * content_scores(users) + w_pop * pop_n[None, :]
    return predict


# AVALIAÇÃO (mesmo protocolo da Etapa 2, com cobertura e popularidade média)


BATCH = 400
EPS = 1e-9

def evaluate(predict_fn, users, targets, ks=(5, 10, 20)):
    ranks, top10 = [], []
    for s in range(0, len(users), BATCH):
        u, t = users[s:s + BATCH], targets[s:s + BATCH]
        S = np.asarray(predict_fn(u), dtype=np.float64) + EPS * pop_n
        c = R_train[u].tocoo()
        S[c.row, c.col] = -1e9
        tv = S[np.arange(len(t)), t]
        ranks.append((S >= tv[:, None]).sum(1))
        top10.append(np.argpartition(-S, 10, axis=1)[:, :10])
    ranks, top10 = np.concatenate(ranks), np.concatenate(top10)

    m = {}
    for k in ks:
        m[f"HR@{k}"] = float((ranks <= k).mean())
        m[f"NDCG@{k}"] = float(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0).mean())
    m["MRR@20"] = float(np.where(ranks <= 20, 1 / ranks, 0).mean())
    m["Cobertura@10"] = len(np.unique(top10)) / N_ITEMS
    m["PopMedia@10"] = float(item_cnt[top10].mean())
    m["_ranks"] = ranks
    return m

def bootstrap_ci(ranks, k=10, n=1000):
    """IC 95% do HR@k por bootstrap."""
    hits = (ranks <= k).astype(float)
    b = rng.choice(hits, size=(n, len(hits))).mean(1)
    return np.percentile(b, [2.5, 97.5])


# 1) LINHA DE BASE: popularidade e PoC da Etapa 2 (Item-kNN, shrink = 20)


t0 = time.time()
resultados = {}
resultados["Popularidade"] = evaluate(lambda u: np.tile(pop_n, (len(u), 1)), test_u, test_t)
S_poc = build_item_knn(R_train, shrink=20)
resultados["Item-kNN (PoC Etapa 2)"] = evaluate(lambda u: (R_train[u] @ S_poc).toarray(), test_u, test_t)


# 2) AJUSTE: busca de hiperparâmetros na VALIDAÇÃO (o teste não é usado aqui)


print("\nBusca do shrinkage do Item-kNN (validação)...")
melhor_shrink, melhor_ndcg = None, -1
for shrink in [0, 5, 10, 20, 50, 100]:
    S = build_item_knn(R_train, shrink=shrink)
    ndcg = evaluate(lambda u: (R_train[u] @ S).toarray(), val_u, val_t)["NDCG@10"]
    print(f"   shrink={shrink:<4} NDCG@10={ndcg:.4f}")
    if ndcg > melhor_ndcg:
        melhor_shrink, melhor_ndcg = shrink, ndcg
S_knn = build_item_knn(R_train, shrink=melhor_shrink)

print("\nBusca dos pesos do híbrido (validação)...")
grid = [(1.0, c, p) for c in [0, 0.1, 0.3, 0.5, 1.0] for p in [0, 0.1, 0.3, 0.5, 1.0]]
melhor_w, melhor_ndcg = None, -1
for w in grid:
    ndcg = evaluate(make_hybrid(S_knn, *w), val_u, val_t)["NDCG@10"]
    if ndcg > melhor_ndcg:
        melhor_w, melhor_ndcg = w, ndcg
print(f"   melhores pesos (kNN, conteúdo, pop) = {melhor_w} | NDCG@10 validação = {melhor_ndcg:.4f}")


# 3) REAVALIAÇÃO NO TESTE


resultados[f"Item-kNN ajustado (shrink={melhor_shrink})"] = evaluate(lambda u: (R_train[u] @ S_knn).toarray(), test_u, test_t)
resultados["Híbrido (kNN + conteúdo + pop)"] = evaluate(make_hybrid(S_knn, *melhor_w), test_u, test_t)

linhas = []
for nome, m in resultados.items():
    lo, hi = bootstrap_ci(m["_ranks"])
    linhas.append({"Modelo": nome,
                   "HR@5": m["HR@5"], "HR@10": m["HR@10"], "HR@20": m["HR@20"],
                   "NDCG@10": m["NDCG@10"], "MRR@20": m["MRR@20"],
                   "IC95% HR@10": f"[{lo * 100:.2f}%, {hi * 100:.2f}%]",
                   "Cobertura@10": m["Cobertura@10"], "PopMedia@10": m["PopMedia@10"]})
tabela = pd.DataFrame(linhas)
tabela.to_csv(OUT_DIR / "metricas_etapa3.csv", index=False, float_format="%.4f")

print("\n===============================================================================")
print("COMPARAÇÃO DOS MODELOS (CONJUNTO DE TESTE)")
print("===============================================================================")
print(tabela.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nTempo total: {time.time() - t0:.1f}s | tabela salva em Resultados/metricas_etapa3.csv")
