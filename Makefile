SHELL := /bin/bash
DC ?= docker compose
COMPOSE_FILE ?= docker-compose.yaml
PROJECT ?= tmdb-mlflow
MLFLOW_SERVICE ?= mlflow
MLFLOW_PORT ?= 5000
BACKUP_DIR ?= mlflow_backups
API_PORT   ?= 8000
API_HOST   ?= 0.0.0.0

.PHONY: help up-mlflow down-mlflow purge-mlflow restart-mlflow ps-mlflow logs-mlflow url-mlflow backup-mlflow restore-mlflow env-mlflow nuke-mlflow run-api run-api-dev up-api down-api up-all down-all

help:
	@echo "Hedefler:"
	@echo "  make up-all          : MLflow + API servislerini birlikte başlatır"
	@echo "  make down-all        : Tüm servisleri durdurur"
	@echo "  make up-api          : Yalnızca API container'ını başlatır"
	@echo "  make down-api        : API container'ını durdurur"
	@echo "  make run-api         : FastAPI inference servisini başlatır (production)"
	@echo "  make run-api-dev     : FastAPI inference servisini --reload ile başlatır (geliştirme)"
	@echo "  make up-mlflow       : MLflow servisini başlatır"
	@echo "  make down-mlflow     : MLflow servisini durdurur"
	@echo "  make purge-mlflow    : MLflow servisini durdurur ve volume'u siler"
	@echo "  make restart-mlflow  : MLflow servisini yeniden başlatır"
	@echo "  make ps-mlflow       : MLflow servisi durumunu gösterir"
	@echo "  make logs-mlflow     : MLflow loglarını takip eder"
	@echo "  make url-mlflow      : MLflow UI URL'ini yazdırır"
	@echo "  make backup-mlflow   : MLflow DB ve artifact'larını yedekler"
	@echo "  make restore-mlflow  : En son yedeği (veya FILE=... ile belirtileni) geri yükler"
	@echo "  make env-mlflow      : MLflow container içindeki env değişkenlerini gösterir"
	@echo "  make nuke-mlflow     : MLflow container, volume ve verilerini tamamen siler"

up-mlflow:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) up -d $(MLFLOW_SERVICE)

down-mlflow:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) stop $(MLFLOW_SERVICE)

purge-mlflow:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) down -v

restart-mlflow:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) restart $(MLFLOW_SERVICE)

ps-mlflow:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) ps $(MLFLOW_SERVICE)

logs-mlflow:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) logs -f $(MLFLOW_SERVICE)

url-mlflow:
	@echo "MLflow UI: http://localhost:$(MLFLOW_PORT)"

backup-mlflow:
	@mkdir -p $(BACKUP_DIR)
	$(eval BACKUP_FILE := $(BACKUP_DIR)/mlflow_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz)
	@$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) exec -T $(MLFLOW_SERVICE) sh -lc \
		"tar -czf - -C / mlflow" \
		> $(BACKUP_FILE)
	@echo "Yedek oluşturuldu: $(BACKUP_FILE)"

restore-mlflow:
	$(eval RESTORE_FILE ?= $(shell ls -t $(BACKUP_DIR)/mlflow_backup_*.tar.gz 2>/dev/null | head -1))
	@test -n "$(RESTORE_FILE)" || { echo "Yedek bulunamadı. FILE=<path> ile belirtin veya önce backup-mlflow çalıştırın."; exit 1; }
	@echo "Geri yükleniyor: $(RESTORE_FILE)"
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) stop $(MLFLOW_SERVICE)
	docker run --rm \
		-v $(PROJECT)_mlflow_data:/mlflow \
		-v $(CURDIR)/$(BACKUP_DIR):/backup \
		alpine sh -c "rm -rf /mlflow/* && tar -xzf /backup/$$(basename $(RESTORE_FILE)) -C /"
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) start $(MLFLOW_SERVICE)
	@echo "Geri yükleme tamamlandı: $(RESTORE_FILE)"

env-mlflow:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) exec $(MLFLOW_SERVICE) env

nuke-mlflow:
	@echo "UYARI: Bu işlem MLflow container, volume ve tüm verileri siler."
	@$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) down -v --remove-orphans

up-all:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) up -d

down-all:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) down

up-api:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) up -d api

down-api:
	$(DC) -f $(COMPOSE_FILE) -p $(PROJECT) stop api

run-api:
	PYTHONPATH=$(CURDIR) uvicorn app.api:app --host $(API_HOST) --port $(API_PORT)

run-api-dev:
	PYTHONPATH=$(CURDIR) uvicorn app.api:app --host $(API_HOST) --port $(API_PORT) --reload
