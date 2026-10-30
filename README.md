# Sistema de Recomendação de Produtos para E-commerce

Projeto Aplicado III — Universidade Presbiteriana Mackenzie (EaD)
Professor: Gustavo Scalabrini Sampaio

## Sobre o projeto

Desenvolvimento de um sistema de recomendação de produtos para e-commerce, baseado em técnicas de Machine Learning, utilizando dados históricos de pedidos e avaliações de clientes. O projeto tem um recorte extensionista voltado a ampliar a visibilidade de pequenos vendedores, dialogando com os ODS 8 (Trabalho decente e crescimento econômico) e ODS 9 (Indústria, inovação e infraestrutura).

## Base de dados

**Brazilian E-Commerce Public Dataset by Olist** (Kaggle)
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Integrantes

| Nome | RA |
|---|---|
| Luana Isabella Xavier Porto | 10443026 |
| Jaques Eler Antunes | 10748092 |
| Felipe Barreto | 10742390 |
| Gabrielle Solange Ferreira | 10414956 |

## Estrutura do repositório

```
/Dataset     -> tabelas CSV da base Olist
/Script      -> Explorato.py (EDA + PoC) e Modelo.py (ajuste e reavaliação)
/Resultados  -> gráficos da EDA e tabela de métricas
/Docs        -> documento do projeto (entregas por etapa)
```

## Etapas do projeto

- [x] Etapa 1: Concepção do Produto
- [x] Etapa 2: Definição do Produto
- [x] Etapa 3: Metodologia
- [ ] Etapa 4: Resultados e Conclusão

## Metodologia

**Dados.** Usamos os pedidos entregues da base Olist, sem os que receberam nota 1 ou 2. Cada compra vira um "1" numa matriz cliente × produto (feedback implícito), com o cliente identificado por `customer_unique_id`.

**Modelo.** Um recomendador híbrido que soma três notas, cada uma com um peso:

| Componente | Ideia |
|---|---|
| Item-kNN (cosseno + *shrinkage*) | "Quem comprou X também comprou Y" |
| Conteúdo por categoria | Produtos das categorias que o cliente já compra |
| Popularidade | Produtos mais vendidos, para quando há pouco histórico |

**Avaliação.** Escondemos a última compra de cada cliente e verificamos se o modelo a coloca no topo da lista (*leave-one-out* temporal). Os parâmetros são escolhidos num grupo de validação (30%) e o resultado final é medido num grupo de teste separado (70%), com HR@K, NDCG@K, MRR, cobertura do catálogo e popularidade média das recomendações.

**Resultado.** O híbrido acerta o produto no top-10 para 8,67% dos clientes de teste, contra 4,97% da prova de conceito da Etapa 2, e recomenda produtos mais variados. Detalhes na seção 4 de `Docs/Etapa3_Projeto_Aplicado_III.docx`.
