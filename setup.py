from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


setup(
    name="longhaul",
    version="0.1.0",
    description="Long Haul by TEI: simple local CLI for preparing and training MLX fine-tunes.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="TEI",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["PyYAML>=6.0"],
    extras_require={"train": ["mlx-lm[train]>=0.31.0"]},
    entry_points={"console_scripts": ["longhaul=longhaul.cli:main"]},
)
