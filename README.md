# CVRP Heuristics vs Machine Learning: Benchmarking Tradeoffs

This repository supports ongoing research on the Capacitated Vehicle Routing Problem (CVRP), focused on modeling classical CVRP heuristics and benchmarking them against modern machine-learning-based approaches. The goal of this is to

- Build and analyze baseline CVRP heuristic methods.
- Compare solution quality, runtime, and scalability against ML-oriented solvers.
- Use established CVRP benchmark sets and prior literature to ground empirical comparisons.

## Usage

```
python3.10 -m venv venv     # On Windows: py -3.10 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## References

- CVRPLib.
- Sirui Li and Zhongxia Yan and Cathy Wu, "Learning to Delegate for Large-scale Vehicle Routing." https://doi.org/10.48550/arXiv.2107.04139
- Florian Arnold, Michel Gendreau, Kenneth Sorensen. "Efficiently solving very large-scale routing problems." *Computers & Operations Research*, Volume 107, July 2019, Pages 32-42. https://doi.org/10.1016/j.cor.2019.03.006
- Roberto Asin-Acha, Alexis Espinoza, Olivier Goldschmidt, Dorit S. Hochbaum, Isaias I. Huerta. "Selecting Fast Algorithms for the Capacitated Vehicle Routing Problem with Machine Learning Techniques."
- Fei Liu, Chengyu Lu, Lin Gui, Qingfu Zhang, Xialiang Tong, Mingxuan Yuan. "Heuristics for Vehicle Routing Problem: A Survey and Recent Advances."

## Acknowledgement

This research is being funded by the [Marco Antonio Firebaugh Scholars Program](). I am very grateful to [Professor Chiwei Yan](https://ieor.berkeley.edu/people/chiwei-yan/) who has supported my throughout the research. I also want to acknowledge [Cathie Wu et. als's paper](https://mit-wu-lab.github.io/learning-to-delegate/) and [Professor Dorit Hochbaum](https://hochbaum.ieor.berkeley.edu/) for being important resources to help me set up the development of this project. I also want to thank the [Open Computing Facility](https://www.ocf.berkeley.edu/) for providing High Performance Computing.