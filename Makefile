CONTAINER = mc
DEXEC = docker exec -it $(CONTAINER)
TEST = $(DEXEC) pytest  --log-cli-level=INFO
ifeq ($(I_AM_INSIDE_DOCKER_CONTAINER),true)
	DEXEC =
endif

# ============== [ Container control ] ==============

init:
	docker-compose up -d --build
sh:
	docker-compose exec -it $(CONTAINER) bash
stop:
	docker-compose stop

# ============== [ Tests ] ==============

test:
	$(TEST)
tests: test

testllm:
	$(TEST) tests/llm_envs
