.PHONY: install install-pipeline seed api dashboard test docker

install:
	pip install -r requirements.txt

install-pipeline:
	pip install -r requirements-pipeline.txt

seed:
	python scripts/seed_demo.py --reset

api:
	uvicorn app.main:app --reload --port 8000

dashboard:
	API_URL=http://localhost:8000 streamlit run dashboard/streamlit_app.py --server.port 8501

test:
	python -m pytest

docker:
	docker compose up --build
