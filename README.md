# Graph Problem Swarm Intelligence

Course project exploring **swarm intelligence** approaches to graph problems (routing, partitioning, community structure) with NetworkX experiments that run in Google Colab or locally.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gopalenparasuraman/graph-problem-swarm-intelligence/blob/main/notebooks/01_swarm_graph_starter.ipynb)

**Colab notebook:**  
https://colab.research.google.com/github/gopalenparasuraman/graph-problem-swarm-intelligence/blob/main/notebooks/01_swarm_graph_starter.ipynb

## Repo layout

| Path | Role |
|------|------|
| `notebooks/01_swarm_graph_starter.ipynb` | Colab-ready smoke demo (NetworkX + trivial swarm placeholder) |
| `requirements.txt` | `networkx`, `matplotlib`, `numpy` |

## Local workflow

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_swarm_graph_starter.ipynb
```

Or open the Colab badge above (no local install).

## Push notes

Remote:

```text
https://github.com/gopalenparasuraman/graph-problem-swarm-intelligence.git
```

First push from this tree (authenticated as a collaborator with write access):

```bash
git remote add origin https://github.com/gopalenparasuraman/graph-problem-swarm-intelligence.git
# if origin exists:
# git remote set-url origin https://github.com/gopalenparasuraman/graph-problem-swarm-intelligence.git
git push -u origin main
```

After `main` is on GitHub, the Colab badge and URL above resolve.
