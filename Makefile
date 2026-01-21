CONTAINER = mc
DEXEC = docker exec -it $(CONTAINER)
TEST = $(DEXEC) python -m pytest  --log-cli-level=INFO
ifeq ($(I_AM_INSIDE_DOCKER_CONTAINER),true)
	DEXEC =
endif

# ============== [ Container control ] ==============

up:
	docker-compose up -d

build:
	docker-compose up -d --build

force-rebuild:
	docker-compose build --no-cache

init:
	make build && make sh
sh:
	$(DEXEC) bash
start:
	docker-compose start
stop:
	docker-compose stop
cs:
	$(DEXEC) flake8 microcore tests examples
black:
	$(DEXEC) black microcore tests examples
# ============== [ Tests ] ==============

test:
	$(TEST)
tests: test

# Test on real API services using .env.test.* files
testapi:
	$(TEST) tests/apis
testapis: testapi
apitest: testapi
apitests: testapi

pkg:
	$(DEXEC) python -m build --wheel --sdist .

publish:
	$(DEXEC) twine upload dist/* -u __token__ -p $(PYPI_TOKEN) --verbose
upload: publish

doc:
	rm -rf docs && pdoc microcore --docformat google -o ./docs

docs: doc

watchdoc:
	$(DEXEC) watchmedo shell-command --patterns="*.py" --recursive --command='make doc' microcore
