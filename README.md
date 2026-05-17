# SDF-GAN Autoresearch

Pipeline de pesquisa automatizada de um modelo de *Stochastic Discount Factor* (SDF) treinado de forma adversarial (GAN), seguindo Chen, Pelger e Zhu (2019) — *"Deep Learning in Asset Pricing"*. Um agente LLM itera sobre `train.py` propondo experimentos, treinando em GPU remota (AWS) e registrando os resultados em `results.tsv`.

## Requisitos

- Python 3.10+ com `pip`
- GPU NVIDIA com driver CUDA 12.1 (para execução local opcional)
- Conta AWS com credenciais configuradas (`aws configure`) — necessário para o loop de experimentos
- `bash` (Git Bash no Windows) para executar os scripts em `aws/`
- Dataset pré-processado em `datasets/` (não versionado)

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/leonlbc/sdfgan-tcc.git
cd sdfgan-tcc

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows (PowerShell)

# 3. Instalar dependências
#    Para GPU NVIDIA (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

#    Para CPU apenas:
pip install -r requirements.txt
```

## Estrutura do projeto

| Arquivo / pasta       | Função                                                              |
|-----------------------|---------------------------------------------------------------------|
| `train.py`            | Script de treinamento — único arquivo modificado pelo agente        |
| `validate.py`         | Avaliação OOS rolling-window (executada manualmente após o loop)    |
| `prepare.py`          | Carregamento de dados, métricas e *evaluation harness* (não editar) |
| `program.md`          | Protocolo completo de experimentação seguido pelo agente            |
| `paper_knowledge.md`  | Notas sobre o paper de referência                                   |
| `results.tsv`         | Registro de todos os experimentos executados                        |
| `aws/`                | Scripts para provisionar GPU AWS e executar jobs remotos            |
| `datasets/`           | Dados pré-processados (não versionado)                              |

## Execução

### Treinamento local (opcional, apenas com GPU)

```bash
python train.py            # seed 42 (padrão)
python train.py 43         # outra seed
```

Ao final, o script imprime um bloco de resumo com `valid_sharpe`, `train_sharpe`, `valid_loss`, `train_loss`, `valid_ev`, `train_ev`, `train_time_s` e `peak_vram_mb`.

### Treinamento na AWS (fluxo recomendado)

Os scripts em `aws/` automatizam o ciclo completo: provisionar instância GPU, sincronizar código, executar o job e baixar os resultados.

```bash
# Provisionar instância (primeira vez ou após terminar a anterior)
bash aws/launch.sh
bash aws/setup.sh

# Enviar dataset (apenas na primeira execução em uma instância nova)
bash aws/upload-data.sh

# Ciclo de cada experimento
bash aws/sync.sh                          # envia o train.py atual
bash aws/run-job.sh train                 # roda python train.py (seed 42)
bash aws/run-job.sh train 43              # roda com seed 43
bash aws/download.sh                      # baixa run.log e artefatos

# Encerrar a instância ao final do dia (evita custo ocioso)
bash aws/terminate.sh
```

### Avaliação out-of-sample (manual)

Após o loop de experimentos terminar, o pesquisador humano executa a avaliação rolling-window:

```bash
bash aws/sync.sh && bash aws/run-job.sh validate && bash aws/download.sh
```

## Loop autônomo do agente

O agente LLM (Claude Code) segue o protocolo descrito em `program.md`:

1. Lê `notes.md` e o estado de `results.tsv`.
2. Escolhe uma categoria (`structure`, `dynamics`, `compression`, `composition`, `follow-up`) e um nó pai.
3. Modifica `train.py`, comita e executa na AWS.
4. Registra a métrica em `results.tsv` e decide se mantém ou descarta o experimento.
5. Repete até a condição de parada (20 descartes consecutivos sem novo *keep*).

Para iniciar o loop, abra o Claude Code neste diretório — ele lerá `CLAUDE.md` e `program.md` automaticamente.

## Visualização dos resultados

```bash
python plot_experiments.py
```

Gera `fig_experiment_trail.png` e `fig_experiment_trail.pdf` com a trilha dos experimentos sobre o `valid_sharpe`.

## Referência

Chen, L., Pelger, M., & Zhu, J. (2019). *Deep Learning in Asset Pricing*. SSRN 3350138.
