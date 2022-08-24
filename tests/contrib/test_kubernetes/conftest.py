import pytest


def _cut_empty_lines(string):
    return "\n".join(line for line in string.splitlines() if line)


@pytest.fixture
def k8s_default_manifest():
    return _cut_empty_lines(
        """apiVersion: v1
kind: Namespace
metadata:
  name: mlem-ml-app
  labels:
    name: mlem-ml-app

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml
  namespace: mlem-ml-app
spec:
  selector:
    matchLabels:
      app: ml
  template:
    metadata:
      labels:
        app: ml
    spec:
      containers:
      - name: ml
        image: ml:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080

---

apiVersion: v1
kind: Service
metadata:
  name: ml
  namespace: mlem-ml-app
  labels:
    run: ml
spec:
  ports:
  - port: 8080
    protocol: TCP
    targetPort: 8080
  selector:
    app: ml
  type: NodePort
"""
    )


@pytest.fixture
def k8s_manifest():
    return _cut_empty_lines(
        """apiVersion: v1
kind: Namespace
metadata:
  name: mlem-hello-app
  labels:
    name: mlem-hello-app

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello
  namespace: mlem-hello-app
spec:
  selector:
    matchLabels:
      app: hello
  template:
    metadata:
      labels:
        app: hello
    spec:
      containers:
      - name: hello
        image: hello:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8080

---

apiVersion: v1
kind: Service
metadata:
  name: hello
  namespace: mlem-hello-app
  labels:
    run: hello
spec:
  ports:
  - port: 8080
    protocol: TCP
    targetPort: 8080
  selector:
    app: hello
  type: LoadBalancer
"""
    )
