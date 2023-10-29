CONTAINER = mc
DEXEC = docker exec -it $(CONTAINER)
TEST = $(DEXEC) python -m pytest  --log-cli-level=INFO
ifeq ($(I_AM_INSIDE_DOCKER_CONTAINER),true)
	DEXEC =
endif

# ============== [ Container control ] ==============

init:
	docker-compose up -d --build && docker-compose exec -it $(CONTAINER) bash
sh:
	docker-compose exec -it $(CONTAINER) bash
stop:
	docker-compose stop

# ============== [ Tests ] ==============

test:
	$(TEST)
tests: test

testapi:
	$(TEST) tests/apis
testapis: testapi
apitest: testapi
apitests: testapi
