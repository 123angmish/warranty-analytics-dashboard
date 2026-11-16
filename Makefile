.PHONY: all data db train export test app clean

all: data db train export test

data:
	python -m src.generate_data

db:
	python -m src.db

train:
	python -m src.classifier

export:
	python -m src.report
	python -m src.export

test:
	python -m pytest tests/ -v

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/*.csv data/*.db models/*.joblib reports/*.json reports/*.xlsx reports/car_reports/*.pdf data/powerbi/*.csv
