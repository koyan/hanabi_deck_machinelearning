# Variables
IMAGE_NAME=hanabi-engine
# We mount the current directory to /app
# We still mount the datasets separately if they are in different locations,
# but if they are inside your project folder, this is even simpler.
DOCKER_RUN=docker run --rm -it \
	-v $(shell pwd):/app \
	$(IMAGE_NAME)

build:
	docker build -t $(IMAGE_NAME) .

bash:
	$(DOCKER_RUN) /bin/bash