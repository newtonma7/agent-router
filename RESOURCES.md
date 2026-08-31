# Contextual Bandits and Reinforcement Learning Resources

## Knowledge

- [Book: _Bandit Algorithms_ by Tor Lattimore and Csaba Szepesvári](https://tor-lattimore.com/downloads/book/book.pdf)
  A rigorous, free reference. Use Chapter 1 for bandit foundations and Chapter 18 for contextual bandits; this is the primary reference for the router's learning formulation.
- [Book: _Reinforcement Learning: An Introduction_, 2nd ed. by Richard Sutton and Andrew Barto](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf)
  The standard broad RL text. Use Chapters 1–2 to distinguish one-step bandits from multi-step RL and to learn return/value terminology.
- [Paper: “Contextual Bandits with Linear Payoff Functions” by Chu et al.](https://proceedings.mlr.press/v15/chu11a.html)
  The canonical LinUCB paper. Use it to connect the implementation's per-action linear model and confidence bonus to the formal algorithm.
- [Paper: “A Contextual-Bandit Approach to Personalized News Article Recommendation” by Li et al.](https://arxiv.org/abs/1003.0146)
  A practical contextual-bandit case study showing context-dependent selection and offline evaluation from logged data.
- [Tutorial: Vowpal Wabbit Contextual Bandits](https://vowpalwabbit.org/tutorials/contextual_bandits.html)
  A practical reference for exploration, logged feedback, and real-world contextual-bandit workflows. Use after the core concepts are clear.

## Wisdom (Communities)

No community preference has been established yet. Prefer discussing experiment design with experienced ML practitioners after the first benchmark results exist.

## Gaps

- A compact, project-specific explanation of reward-constrained routing will be developed in a later lesson.
