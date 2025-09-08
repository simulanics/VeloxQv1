# Contributing to VeloxQ-Style QUBO Solver (POC)

First off, thank you for considering contributing! 🎉  
This project is an independent **proof-of-concept reconstruction** of a VeloxQ-style QUBO solver, inspired by public descriptions of **VeloxQ 1 by Quantumz.io (Poland)**.  
It is not affiliated with, endorsed by, or derived from proprietary technology. All contributions should keep that spirit clear.

---

## How to Contribute

We welcome:

- 🐛 **Bug reports** and fixes
- 💡 **Feature suggestions** (e.g., new heuristics, HUBO reduction helpers, sparse support)
- 📖 **Documentation improvements**
- 🔬 **Benchmarks** comparing this POC against other solvers

### 1. Fork and branch

1. Fork this repository  
2. Create a branch for your changes:

```bash
git checkout -b feature/my-new-idea
````

### 2. Code style

* Use **Python 3.9+**
* Follow **PEP8** as much as practical
* Keep functions **self-contained and well-documented**
* Prefer **NumPy** vectorization when possible

### 3. Testing

* Add test cases in `tests/`
* Use `pytest` for consistency
* Always test small QUBO instances with `--exact` to confirm correctness

Run tests locally:

```bash
pytest -v
```

### 4. Commit messages

Use clear, descriptive commit messages:

```
fix: correct ΔE calculation for flip updates
feat: add HUBO → QUBO reduction utility
docs: improve README installation section
```

### 5. Pull requests

* Open a PR against the `main` branch
* Describe your change and why it matters
* If adding features, include at least one example or test
* Be respectful in discussions — this is an educational project

---

## Community Guidelines

* This is a **learning-first, open-source repo** — ideas, critiques, and extensions are encouraged.
* Do **not** submit code copied from proprietary implementations.
* Always credit external inspirations, papers, or projects in PRs.

---

## Attribution

* **Inspiration:** [Quantumz.io](https://quantumz.io) (Poland) for VeloxQ 1
* **Reconstruction & POC Author:** Matthew Combatti

---

## License

By contributing, you agree that your contributions will be licensed under the **MIT License** (see [LICENSE](LICENSE)).

---

💡 *Tip:* Check out [README.md](README.md) for an overview of architecture and usage before diving into code!
