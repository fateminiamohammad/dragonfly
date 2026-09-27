# Build from the repository root so the shared .proto is in the context:
#   docker build -f plugins/example-grpc/Dockerfile -t dragonfly-plugin-policy .
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir grpcio==1.66.2 grpcio-tools==1.66.2
COPY proto ./proto
RUN python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. dragonfly/plugin/v1/plugin.proto \
    && touch dragonfly/__init__.py dragonfly/plugin/__init__.py dragonfly/plugin/v1/__init__.py
COPY plugins/example-grpc/server.py .
RUN useradd --system plugin
USER plugin
EXPOSE 50051
CMD ["python", "server.py"]
