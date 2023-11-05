CONTAINER = mc
DEXEC = docker exec -it $(CONTAINER)
TEST = $(DEXEC) python -m pytest  --log-cli-level=INFO
ifeq ($(I_AM_INSIDE_DOCKER_CONTAINER),true)
	DEXEC =
endif

# ============== [ Container control ] ==============

init:
	docker-compose up -d --build && $(DEXEC) bash
sh:
	$(DEXEC) bash
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

#doc:
#	rm -rf docs/microcore && pdoc microcore --html --output-dir ./docs -f
doc:
	rm -rf docs && pdoc microcore --docformat google -o ./docs

docs: doc

watchdoc:
	$(DEXEC) watchmedo shell-command --patterns="*.py" --recursive --command='make doc' microcore
