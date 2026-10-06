PY ?= python

.PHONY: fetch check data index graph dense eval app test all sample crosswalk

fetch:
	$(PY) scripts/00_fetch_data.py

crosswalk:
	$(PY) -m kanoon_bridge.ingest.parse_crosswalk

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

eval:
	$(PY) scripts/05_run_all_evals.py

app:
	streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

all: data index graph eval

# synthetic end-to-end run (no Hugging Face access needed; BNS source still required)
sample:
	$(PY) tests/data/make_ilpcsr_sample.py
	$(PY) scripts/01_build_corpus.py --sample
	$(PY) scripts/02_build_index.py
	$(PY) scripts/03_build_graph.py --sample
