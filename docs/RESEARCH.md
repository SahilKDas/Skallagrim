# Existing engine architectures

This comparison concerns the four pinned repositories in `benchmarks/opponents.lock.json`, inspected on October 9, 2026. It is a source review, not a ranking of playing strength.

| Engine | Neural architecture | Search | Checkpoint used here |
| --- | --- | --- | --- |
| [TaflZero](https://github.com/sovelin/taflzero) | Residual CNN with policy/value heads; ONNX inference | Batched MCTS | Repository-provided `default_nn.onnx` |
| [Nichols experimental](https://github.com/nicholasg1997/hnefatafl/tree/experimental) | Four 64-channel convolutional layers with policy/value heads | Neural MCTS | Our labeled upstream-compatible warm start |
| [LuizzK](https://github.com/LuizzK/hnefatafl-engine) | Residual CNN with policy/value heads | PUCT-style MCTS; this pinned implementation expands with uniform priors | Our labeled 16-channel, two-block warm start |
| [OpenTafl](https://github.com/jslater89/OpenTafl) | Handcrafted evaluator | Alpha-beta with transposition table and move ordering | Native evaluator |

The source review supports CNN plus MCTS as an existing tafl approach. These four implementations do not provide a ready-made NNUE, GNN or Transformer implementation for Skallagrim. That observation is limited to this set; it is not a claim that no such tafl research exists elsewhere.

Skallagrim therefore implements its own NNUE feature representation and incremental accumulator for Iota/Beta, while Rho follows the established policy/value CNN plus MCTS pattern. Iota and Beta share an evaluator to make search differences measurable. Beta uses uniform priors because a value-only NNUE does not supply a learned move policy.

Architecture alone does not determine strength. Copenhagen rules compatibility, training data, role coverage, inference cost and search under the actual time limit all need measurement. Our initial models use a minimum-sized baseline bootstrap; the Python opponents' warm starts are not author-published trained agents. See [training provenance](TRAINING.md), [rule limitations](RULES.md), and [benchmark conditions](BENCHMARKS.md).
