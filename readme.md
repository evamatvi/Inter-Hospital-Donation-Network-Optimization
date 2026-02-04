# Inter-Hospital Donation Network Optimization

This project addresses the problem of coordinating organ donations between hospitals at an international level.

The system considers 40 hospitals that must be assigned to 4 exclusive donation networks, with each hospital belonging to exactly one network. The goal is to find an optimal assignment that minimizes the distance between hospitals within the same network while also considering the similarity of their population characteristics.

## Approach

The problem is solved using local search algorithms. Each state represents a possible assignment of the 40 hospitals to the 4 networks.

The following algorithms have been implemented and compared:
- Local Beam Search
- Hill Climbing
- Simulated Annealing
- Random Search (as a baseline)

Probabilistic inference techniques are also used to work with Bayesian networks and handle incomplete data.

## Project Structure

- **bn.py**: Basic implementation of Bayesian networks, factors, and probability distributions.
- **inferencia.py**: Inference algorithms, including Variable Elimination, Rejection Sampling, and Weighted Sampling.
- **p1.py**: Execution of experiments and implementation of local search methods.
- **my_bns.py**: Definitions of the Bayesian networks used in the experiments.
- **data.csv**: Dataset containing hospital information used in the experiments.

## Stopping Criteria

Local search algorithms (`hill_climbing`, `simulated_annealing`, `local_beam_search`) use stopping criteria based on improvement tolerance. If no significant improvement is observed, the algorithm may reach the maximum number of iterations without further progress.

## Performance

Execution time varies depending on parameter configuration, such as the number of iterations and the beam size.

## Autors

Eva Matabosch 

Iman Tarfass



