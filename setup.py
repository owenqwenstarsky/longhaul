from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


setup(
    name="longhaul",
    version="0.1.0",
    description="Long Haul by TEI: simple local CLI for preparing and training MLX fine-tunes.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/owenqwenstarsky/longhaul",
    license="MIT",
    python_requires=">=3.9",
    keywords=["cli", "fine-tuning", "mlx", "llm", "training", "qwen"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Build Tools",
    ],
    project_urls={
        "Homepage": "https://github.com/owenqwenstarsky/longhaul",
        "Repository": "https://github.com/owenqwenstarsky/longhaul",
        "Issues": "https://github.com/owenqwenstarsky/longhaul/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["PyYAML>=6.0"],
    extras_require={"train": ["mlx-lm[train]>=0.31.0"]},
    entry_points={"console_scripts": ["longhaul=longhaul.cli:main"]},
)
