reset:
	rm -rf .venv __pypackages__ uv.lock rockdaisy.egg-info
	uv venv --python=python3.12
	. .venv/bin/activate && uv pip install -e .
