"""Delay state copies until a Luizz child is visited; retain upstream PUCT logic.

The original eagerly makes every legal child, exceeding a 300-ms budget even
in one simulation. This adapter retains priors, visit order, values and backups.
"""


def install(module):
    original = module.MCTSNode.expand
    base = module.MCTSNode

    class LazyNode(base):
        @property
        def game_state(self):
            if self._state is None:
                self._state = self.parent.game_state.copy()
                self._state.make_move(self.move)
            return self._state

        @game_state.setter
        def game_state(self, state):
            self._state = state

    def expand(node, policy_probs, legal_moves):
        if node.is_expanded:
            return
        for move in legal_moves:
            node.children[move] = LazyNode(
                None, parent=node, prior_prob=1.0 / len(legal_moves), move=move
            )
        node.is_expanded = True

    base.expand = expand
    return original
