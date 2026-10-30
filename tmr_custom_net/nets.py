"""
nets.py - Le tue reti: definizione, dataset, training. Il notebook FINN riceve da qui la rete GIA' ADDESTRATA.

Per aggiungere una rete:
  1. definisci il suo nn.Module (Brevitas, se deve andare su FINN);
  2. scrivi una sottoclasse di `NetSpec` che sa costruirla, addestrarla, valutarla e dare un'immagine di esempio;
  3. registrala in `NETS`.
Poi nel notebook basta cambiare `NET_NAME`. Il notebook non conosce ne' la rete ne' il dataset.
"""
from __future__ import annotations

import os
from shutil import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU


# --------------------------------------------------------------------------------------------
# Contratto che il notebook si aspetta da ogni rete
# --------------------------------------------------------------------------------------------
class NetSpec:
    """Descrive una rete per il notebook. Sovrascrivi i metodi (tranne `copy_validation_data`, opzionale)."""

    input_shape: Tuple[int, ...] = (1, 1, 28, 28)   # (batch, canali, altezza, larghezza)

    def build(self, wbits: int, abits: int) -> nn.Module:
        """Rete NON addestrata, con costruttore senza altri argomenti."""
        raise NotImplementedError

    def train(self, net: nn.Module, device: str, epochs: int, lr: float, batch_size: int) -> None:
        """Addestra `net` sul posto."""
        raise NotImplementedError

    def evaluate(self, net: nn.Module, device: str) -> float:
        """Accuratezza (0..1) sul test set."""
        raise NotImplementedError

    def sample(self) -> Tuple[np.ndarray, int]:
        """Un'immagine di test (uint8, qualsiasi forma con gli stessi elementi di input_shape) e la sua etichetta.
        Finisce nel pacchetto di deploy come input.npy / expected_label.txt."""
        raise NotImplementedError

    def copy_validation_data(self, deploy_dir: str) -> None:
        """Opzionale: copia nel pacchetto di deploy i dati che lo script di validazione sulla board deve trovare."""


def get_trained_net(name: str, wbits: int, abits: int, ckpt_dir: str, device: str = 'cpu',
                    epochs: int = 5, lr: float = 1e-3, batch_size: int = 128):
    """Restituisce (rete addestrata su CPU in eval, NetSpec).
    Se esiste `<ckpt_dir>/model.pth` lo carica (cancellalo per riaddestrare), altrimenti addestra e lo salva."""
    if name not in NETS:
        raise KeyError(f'rete {name!r} non registrata in NETS. Disponibili: {sorted(NETS)}')
    spec = NETS[name]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = f'{ckpt_dir}/model.pth'
    net = spec.build(wbits, abits).to(device)
    if os.path.isfile(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device))
        print(f'Checkpoint caricato: {ckpt}')
    else:
        spec.train(net, device, epochs, lr, batch_size)
        torch.save(net.state_dict(), ckpt)
        print(f'Checkpoint salvato: {ckpt}')
    print(f'Accuratezza test: {spec.evaluate(net, device) * 100:.2f}%')
    return net.cpu().eval(), spec


# --------------------------------------------------------------------------------------------
# SmallCNN + MNIST (la rete del notebook originale)
# --------------------------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, wbits: int = 4, abits: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            QuantConv2d(1, 16, 3, bias=False, weight_bit_width=wbits),  nn.BatchNorm2d(16), QuantReLU(bit_width=abits),
            QuantConv2d(16, 16, 3, bias=False, weight_bit_width=wbits), nn.BatchNorm2d(16), QuantReLU(bit_width=abits),
            nn.MaxPool2d(2),
            QuantConv2d(16, 32, 3, bias=False, weight_bit_width=wbits), nn.BatchNorm2d(32), QuantReLU(bit_width=abits),
            nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            QuantLinear(800, 64, bias=False, weight_bit_width=wbits), nn.BatchNorm1d(64), QuantReLU(bit_width=abits),
            QuantLinear(64, 10, bias=False, weight_bit_width=wbits))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class MNISTSpec(NetSpec):
    """SmallCNN su MNIST. Se MNIST e' gia' in ./data/MNIST/raw lo usa, altrimenti lo scarica li'."""
    input_shape = (1, 1, 28, 28)
    RAW = './data/MNIST/raw'
    FILES = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    def __init__(self):
        self._train_ds = self._test_ds = None

    def _datasets(self):
        if self._test_ds is None:
            import torchvision
            have = all(os.path.isfile(f'{self.RAW}/{f}') for f in self.FILES)
            print('MNIST già presente' if have else f'MNIST non trovato: lo scarico in {self.RAW}')
            tf = torchvision.transforms.ToTensor()
            self._train_ds = torchvision.datasets.MNIST('./data', train=True, download=not have, transform=tf)
            self._test_ds = torchvision.datasets.MNIST('./data', train=False, download=not have, transform=tf)
        return self._train_ds, self._test_ds

    def build(self, wbits, abits):
        return SmallCNN(wbits, abits)

    def evaluate(self, net, device):
        _, test_ds = self._datasets()
        loader = torch.utils.data.DataLoader(test_ds, batch_size=1000)
        net.to(device).eval()
        ok = 0
        with torch.no_grad():
            for x, y in loader:
                ok += (net(x.to(device)).argmax(1).cpu() == y).sum().item()
        return ok / len(test_ds)

    def train(self, net, device, epochs, lr, batch_size):
        train_ds, _ = self._datasets()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            net.train()
            for x, y in loader:
                opt.zero_grad()
                nn.functional.cross_entropy(net(x.to(device)), y.to(device)).backward()
                opt.step()
            sched.step()
            print(f'epoca {ep + 1}/{epochs}  acc {self.evaluate(net, device) * 100:.2f}%')

    def sample(self):
        _, test_ds = self._datasets()
        return test_ds.data[0].numpy().astype(np.uint8), int(test_ds.targets[0])

    def copy_validation_data(self, deploy_dir):
        # MNIST dentro il pacchetto (cartella mnist/): validate.py lo trova senza scaricare nulla
        os.makedirs(f'{deploy_dir}/mnist', exist_ok=True)
        for f in self.FILES:
            copy(f'{self.RAW}/{f}', f'{deploy_dir}/mnist/')


# --------------------------------------------------------------------------------------------
# Registro: il notebook sceglie la rete per nome (NET_NAME)
# --------------------------------------------------------------------------------------------
NETS = {
    'SmallCNN': MNISTSpec(),
    # 'MiaRete': MiaReteSpec(),
}
