.PHONY: up down logs ps build up-scale bench

up:
	docker compose -f infra/docker-compose/docker-compose.yml up --build -d

up-scale:
	docker compose -f infra/docker-compose/docker-compose.yml -f infra/docker-compose/docker-compose.scale.yml up --build -d

down:
	docker compose -f infra/docker-compose/docker-compose.yml down

logs:
	docker compose -f infra/docker-compose/docker-compose.yml logs -f --tail=200

ps:
	docker compose -f infra/docker-compose/docker-compose.yml ps

build:
	docker compose -f infra/docker-compose/docker-compose.yml build

bench:
	python scripts/benchmark_baseline_vs_milp.py --limit 20 --traffic-multiplier 1.2
