#=============================================================================
# Makefile for testing and running the alldata
#
#=============================================================================
CMD ?= /bin/bash

# check if the user is alan
ifeq ($(USER),root)
	$(error do not run this Makefile as root!)
else
	DEV_CONTAINER := nf-isam
endif

# Disable verbosity
MAKEFLAGS += --silent

# Set default target
.DEFAULT_GOAL := all

DOCKER := $(shell which docker)
DOCKER_COMPOSE := $(shell which docker-compose)
DOCKER_COMPOSE_VERSION := 1.27.4

#=============================================================================
# Development rules


up:
	$(DOCKER_COMPOSE) up --detach \
		--no-color \
		--quiet-pull \
		--remove-orphans \
		${DEV_CONTAINER}
.PHONY: up

stop:
	$(DOCKER_COMPOSE) stop \
		${DEV_CONTAINER}
.PHONY: stop

down:
	$(DOCKER_COMPOSE) down \
		--remove-orphans \
		--volumes \
		--rmi local
.PHONY: down

shell:
	${DOCKER_COMPOSE} exec ${DEV_CONTAINER} ${CMD}
.PHONY: shell

shell-root:
	${DOCKER_COMPOSE} exec -u root ${DEV_CONTAINER} ${CMD}
.PHONY: shell-root
