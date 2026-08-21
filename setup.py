from setuptools import setup, find_packages

setup(
    name="prox",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0,<4.0.0",
        "numpy>=1.20.0,<3.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        "pm4py>=2.7.0,<3.0.0",
        "streamlit>=1.30.0,<2.0.0",
        "reportlab>=4.0.0,<6.0.0",
    ],
    extras_require={
        # BigQuery live data source (docs/dev_roadmap.md). Pulls in
        # first-order-engine's foe.data.DataEngine, gated separately behind
        # its own [bigquery] extra since it's not on PyPI.
        "bigquery": [
            "foe[bigquery] @ git+https://github.com/BasLinders/first-order-engine.git",
        ],
    },
    python_requires=">=3.9",
)
