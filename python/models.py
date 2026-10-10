"""Architectures matching Rust scalar inference and versioned JSON exports."""

import json
from pathlib import Path
import torch
from torch import nn

SQUARES = 121
FEATURES = 364


def encode(fen, history=None):
    board, side = fen.split()
    x = torch.zeros(6, 11, 11)
    for y, row in enumerate(board.split("/")):
        col, digits = 0, ""
        for char in row + "!":
            if char.isdigit():
                digits += char
                continue
            if digits:
                col += int(digits)
                digits = ""
            if char == "!":
                break
            x["adk".index(char), y, col] = 1
            col += 1
    x[3].fill_(float(side == "a"))
    for y, col in [(0, 0), (0, 10), (10, 0), (10, 10), (5, 5)]:
        x[4, y, col] = 1
    x[5].fill_(min((history or [fen]).count(fen), 2) / 2)
    f = torch.cat((x[:3].flatten(), torch.tensor([float(side == "a")])))
    return x, f


def square(s):
    return (int(s[1:]) - 1) * 11 + ord(s[0]) - ord("a")


def policy_index(move):
    split = next(i for i in range(1, len(move)) if move[i].isalpha())
    a, b = square(move[:split]), square(move[split:])
    dx, dy = b % 11 - a % 11, b // 11 - a // 11
    direction, distance = (
        (0, dx) if dx > 0 else (1, -dx) if dx < 0 else (2, dy) if dy > 0 else (3, -dy)
    )
    return (direction * 10 + distance - 1) * 121 + a


class NNUE(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.input = nn.Linear(FEATURES, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.output(self.input(x).clamp(0, 1)).tanh().squeeze(-1)

    def export(self, path):
        save(
            path,
            dict(
                format="skallagrim-nnue-v1",
                hidden=self.hidden,
                weights=self.input.weight.detach().t().contiguous().flatten().tolist(),
                bias=self.input.bias.detach().tolist(),
                output=self.output.weight.detach().flatten().tolist(),
                output_bias=self.output.bias.item(),
            ),
        )


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3, padding=1)
        self.second = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return (x + self.second(self.first(x).relu())).relu()


def conv_export(layer):
    return dict(
        input=layer.in_channels,
        output=layer.out_channels,
        kernel=layer.kernel_size[0],
        weights=layer.weight.detach().flatten().tolist(),
        bias=layer.bias.detach().tolist(),
    )


class CNN(nn.Module):
    def __init__(self, channels=8, blocks=1):
        super().__init__()
        self.channels = channels
        self.stem = nn.Conv2d(6, channels, 3, padding=1)
        self.blocks = nn.ModuleList(Residual(channels) for _ in range(blocks))
        self.policy = nn.Conv2d(channels, 40, 1)
        self.value = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.stem(x).relu()
        for block in self.blocks:
            x = block(x)
        return self.value(x.mean(dim=(2, 3))).tanh().squeeze(-1), self.policy(
            x
        ).flatten(1)

    def export(self, path):
        save(
            path,
            dict(
                format="skallagrim-cnn-v1",
                channels=self.channels,
                stem=conv_export(self.stem),
                blocks=[
                    [conv_export(b.first), conv_export(b.second)] for b in self.blocks
                ],
                policy=conv_export(self.policy),
                value=self.value.weight.detach().flatten().tolist(),
                value_bias=self.value.bias.item(),
            ),
        )


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf8")
