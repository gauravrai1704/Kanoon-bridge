PY ?= python

.PHONY: ltr fetch check data index graph dense eval app test all sample crosswalk compare inspect eval-quick eval-sample plots

fetch:
	$(PY) scripts/00_fetch_data.py

crosswalk:
	$(PY) -m kanoon_bridge.ingest.parse_crosswalk

# cross-check the crosswalk against the government PDF (download it by hand first, see SETUP_AND_RUN.md)
compare:
	$(PY) -m kanoon_bridge.ingest.parse_crosswalk --compare

# print IL-PCSR schema and one example row (after make fetch)
inspect:
	$(PY) -m kanoon_bridge.ingest.load_ilpcsr

check:
	$(PY) scripts/00_check_access.py

data:
	$(PY) scripts/01_build_corpus.py

index:
	$(PY) scripts/02_build_index.py

graph:
	$(PY) scripts/03_build_graph.py

dense:
	$(PY) scripts/04_encode_dense.py

ltr:
	$(PY) scripts/06_train_ltr.py

eval:
	$(PY) scripts/05_run_all_evals.py

eval-quick:
	$(PY) scripts/05_run_all_evals.py --limit 50

eval-sample:
	$(PY) scripts/05_run_all_evals.py --sample --rag-generator extractive

plots:
	$(PY) scripts/05_run_all_evals.py --only-plots

app:
	streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

all: data index graph ltr eval

# synthetic end-to-end run (no Hugging Face access needed; BNS source still required)
sample:
	$(PY) tests/data/make_ilpcsr_sample.py
	$(PY) scripts/01_build_corpus.py --sample
	$(PY) scripts/02_build_index.py
	$(PY) scripts/03_build_graph.py --sample
	$(PY) scripts/06_train_ltr.py --sample --folds 2
