"""
tmr_layers.py - Triplicazione (Triple Modular Redundancy) di layer di una rete PyTorch/Brevitas.

Dati una rete (nn.Module) e i layer da triplicare, `apply_tmr` restituisce una COPIA della rete
in cui ogni layer scelto e' sostituito da un `TMRBlock`: tre repliche indipendenti (parametri non
condivisi) alimentate dallo stesso ingresso. L'originale non viene toccato.

Cosa fa (fase 1):  triplicare i layer scelti e decidere cosa succede alle tre uscite.
Cosa NON fa (fase 2): il voter hardware e la logica di rilevamento/segnalazione guasti.

Le tre uscite si gestiscono con la strategia `merge`:
  'tap'    la replica 0 prosegue nella rete; le uscite delle repliche 1 e 2 vengono "spillate" come
           uscite extra del grafo (TMRExportWrapper), cosi' le tre repliche restano vive nel grafo
           esportato. E' il punto dove in fase 2 si collega il voter. Il forward normale (training,
           accuratezza) restituisce solo l'uscita principale.
  'median' voter software (mediana a 3 = maggioranza quando almeno due repliche concordano).
           Serve per simulare il TMR in PyTorch e iniettare guasti; NON e' sintetizzabile con i layer
           HW di FINN, quindi non va usato nel flusso FINN.

Selezione dei layer (`layers`): lista di "unita'" da triplicare, ciascuna una di
  - int            indice di un layer logico (vedi `logical_layers(net)`): conv/linear + BN + attivazione
  - str            percorso di un singolo modulo, es. 'features.3'
  - tuple/list     blocco di moduli consecutivi dello stesso contenitore, triplicato come un'unica unita'
                   (nessun voto interno), es. ('features.3', 'features.4', 'features.5')
Una singola unita' puo' essere passata anche senza lista (`layers=1`, `layers='features.3'`).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

N_REPLICAS = 3
MERGE_MODES = ('tap', 'median')

LayerSpec = Union[int, str, Sequence[str]]

# --------------------------------------------------------------------------------------------
# Scoperta dei layer logici
# --------------------------------------------------------------------------------------------
_COMPUTE = (nn.modules.conv._ConvNd, nn.Linear)
_NORM = (nn.modules.batchnorm._BatchNorm,)
_POOL_ETC = (nn.modules.pooling._MaxPoolNd, nn.modules.pooling._AvgPoolNd,
             nn.modules.pooling._AdaptiveAvgPoolNd, nn.Flatten, nn.Dropout, nn.Identity)
_TORCH_ACT = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.Hardtanh, nn.Sigmoid, nn.Tanh)


def _is_activation(m: nn.Module) -> bool:
    if isinstance(m, _TORCH_ACT):
        return True
    # attivazioni quantizzate Brevitas (QuantReLU, QuantIdentity, ...): non sono nn.ReLU
    return type(m).__module__.startswith('brevitas') and not isinstance(m, _COMPUTE)


def _is_known(m: nn.Module) -> bool:
    return isinstance(m, _COMPUTE + _NORM + _POOL_ETC) or _is_activation(m) or isinstance(m, TMRBlock)


def _walk(module: nn.Module, prefix: str = '') -> Iterator[Tuple[str, nn.Module]]:
    """Moduli 'foglia' in ordine di registrazione. Non scende dentro conv/linear/BN/attivazioni
    (i quantizzatori Brevitas sono figli di QuantConv2d: non sono layer a se')."""
    for name, child in module.named_children():
        full = f'{prefix}{name}'
        if _is_known(child) or not any(True for _ in child.children()):
            yield full, child
        else:
            yield from _walk(child, full + '.')


def _parent_of(name: str) -> str:
    return name.rsplit('.', 1)[0] if '.' in name else ''


@dataclass(frozen=True)
class LayerBlock:
    index: int
    names: Tuple[str, ...]
    description: str

    def __str__(self) -> str:
        return f'[{self.index}] {", ".join(self.names)}  ->  {self.description}'


def _describe(m: nn.Module) -> str:
    if isinstance(m, nn.modules.conv._ConvNd):
        return f'{type(m).__name__}({m.in_channels}->{m.out_channels}, k={m.kernel_size[0]})'
    if isinstance(m, nn.Linear):
        return f'{type(m).__name__}({m.in_features}->{m.out_features})'
    return type(m).__name__


def logical_layers(net: nn.Module) -> List[LayerBlock]:
    """Raggruppa i moduli in layer logici: un conv/linear piu' l'eventuale BatchNorm e
    attivazione che lo seguono subito dopo (stesso contenitore). Pooling e flatten restano fuori.
    Assume che l'ordine di registrazione dei moduli coincida con l'ordine di esecuzione."""
    groups: List[List[Tuple[str, nn.Module]]] = []
    cur: Optional[List[Tuple[str, nn.Module]]] = None
    for name, m in _walk(net):
        if isinstance(m, _COMPUTE):
            cur = [(name, m)]
            groups.append(cur)
        elif cur is not None and _parent_of(name) == _parent_of(cur[0][0]) and \
                (isinstance(m, _NORM) or _is_activation(m)):
            cur.append((name, m))
        else:
            cur = None
    return [LayerBlock(i, tuple(n for n, _ in g),
                       ' + '.join(_describe(m) for _, m in g))
            for i, g in enumerate(groups)]


# --------------------------------------------------------------------------------------------
# Voter / strategie di merge
# --------------------------------------------------------------------------------------------
def median3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Mediana elemento per elemento di tre tensori, scritta con soli min/max
    (equivale al voto a maggioranza quando almeno due repliche coincidono)."""
    return torch.maximum(torch.minimum(a, b), torch.minimum(torch.maximum(a, b), c))


class TMRBlock(nn.Module):
    """Tre repliche indipendenti di `layer`, alimentate dallo stesso ingresso."""

    def __init__(self, layer: nn.Module, merge: str = 'tap', tmr_name: str = 'layer'):
        super().__init__()
        if merge not in MERGE_MODES:
            raise ValueError(f"merge deve essere uno tra {MERGE_MODES}, non {merge!r}")
        self.merge = merge
        self.tmr_name = tmr_name
        # deepcopy: nessun parametro condiviso tra le repliche
        self.replicas = nn.ModuleList([copy.deepcopy(layer) for _ in range(N_REPLICAS)])
        self._collect: Optional[list] = None   # riempito da TMRExportWrapper

    def forward(self, x):
        outs = [r(x) for r in self.replicas]
        if self.merge == 'median':
            return median3(*outs)
        if self._collect is not None:
            self._collect.extend((f'{self.tmr_name}_r{i}', o) for i, o in enumerate(outs) if i > 0)
        return outs[0]

    def extra_repr(self) -> str:
        return f'name={self.tmr_name}, merge={self.merge}, replicas={N_REPLICAS}'


def tmr_blocks(net: nn.Module) -> List[Tuple[str, TMRBlock]]:
    return [(n, m) for n, m in net.named_modules() if isinstance(m, TMRBlock)]


class TMRExportWrapper(nn.Module):
    """Da usare SOLO per l'export ONNX. Restituisce (uscita_principale, *uscite_tap) cosi' le
    repliche 1 e 2 dei blocchi in modalita' 'tap' diventano uscite del grafo e non vengono potate.
    Senza blocchi 'tap' restituisce la sola uscita principale (export identico a quello normale).
    Dopo un forward/export, `tap_names` contiene i nomi delle uscite extra, nell'ordine."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.tap_names: List[str] = []
        self._blocks = [b for _, b in tmr_blocks(net) if b.merge == 'tap']

    def forward(self, x):
        sink: list = []
        for b in self._blocks:
            b._collect = sink
        try:
            main = self.net(x)
        finally:
            for b in self._blocks:
                b._collect = None
        self.tap_names = [n for n, _ in sink]
        return (main, *[t for _, t in sink]) if sink else main


# --------------------------------------------------------------------------------------------
# apply_tmr
# --------------------------------------------------------------------------------------------
@dataclass
class TMRUnit:
    name: str                 # nome breve (usato per nomi file e tensori)
    modules: Tuple[str, ...]  # percorsi dei moduli triplicati
    description: str
    params_per_replica: int


@dataclass
class TMRReport:
    merge: str
    units: List[TMRUnit] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.units)

    @property
    def tag(self) -> str:
        """Etichetta per cartelle/file, es. 'tmr-tap__features_3'. 'baseline' se nessun TMR."""
        if not self.units:
            return 'baseline'
        return f'tmr-{self.merge}__' + '+'.join(u.name for u in self.units)

    def summary(self) -> str:
        if not self.units:
            return 'TMR: nessun layer triplicato (rete originale)'
        lines = [f'TMR: {len(self.units)} unita\' triplicate (merge={self.merge}, {N_REPLICAS} repliche)']
        for u in self.units:
            lines.append(f'  - {u.name}: {", ".join(u.modules)}  [{u.description}, '
                         f'{u.params_per_replica} parametri/replica]')
        return '\n'.join(lines)


def _unit_name(names: Tuple[str, ...]) -> str:
    first = names[0].replace('.', '_')
    if len(names) == 1:
        return first
    m = re.search(r'(\d+)$', names[-1])
    return f'{first}-{m.group(1)}' if m else f'{first}-{names[-1].replace(".", "_")}'


def resolve_layers(net: nn.Module, layers) -> List[Tuple[str, ...]]:
    """Converte la specifica `layers` in una lista di tuple di percorsi di moduli."""
    if layers is None:
        return []
    if isinstance(layers, (int, str)):
        layers = [layers]
    blocks = None
    units: List[Tuple[str, ...]] = []
    for spec in layers:
        if isinstance(spec, bool):
            raise TypeError('spec di layer non valida: bool')
        if isinstance(spec, int):
            blocks = blocks or logical_layers(net)
            if not 0 <= spec < len(blocks):
                raise IndexError(f'layer logico {spec} inesistente: la rete ne ha {len(blocks)}\n'
                                 + '\n'.join(map(str, blocks)))
            units.append(blocks[spec].names)
        elif isinstance(spec, str):
            units.append((spec,))
        elif isinstance(spec, (tuple, list)) and spec and all(isinstance(s, str) for s in spec):
            units.append(tuple(spec))
        else:
            raise TypeError(f'spec di layer non valida: {spec!r} (attesi int, str o tuple di str)')

    seen = set()
    for names in units:
        for n in names:
            try:
                net.get_submodule(n)
            except AttributeError:
                raise KeyError(f'modulo {n!r} non trovato nella rete') from None
            parts = n.split('.')
            if any(isinstance(net.get_submodule('.'.join(parts[:i])), TMRBlock)
                   for i in range(1, len(parts) + 1)):
                raise ValueError(f'{n!r} e\' gia\' triplicato (o dentro un TMRBlock)')
            if n in seen:
                raise ValueError(f'il modulo {n!r} compare in piu\' unita\' da triplicare')
            if any(n.startswith(s + '.') or s.startswith(n + '.') for s in seen):
                raise ValueError(f'{n!r} e\' annidato in un altro modulo gia\' selezionato')
            seen.add(n)
        if len(names) > 1:   # blocco: stesso contenitore, posizioni consecutive, in ordine
            parents = {_parent_of(n) for n in names}
            if len(parents) != 1:
                raise ValueError(f'i moduli {names} non sono nello stesso contenitore')
            keys = list(net.get_submodule(parents.pop())._modules)
            pos = [keys.index(n.rsplit('.', 1)[-1]) for n in names]
            if pos != list(range(pos[0], pos[0] + len(pos))):
                raise ValueError(f'i moduli {names} non sono consecutivi e in ordine')
    return units


def apply_tmr(net: nn.Module, layers=None, merge: str = 'tap') -> Tuple[nn.Module, TMRReport]:
    """Restituisce (copia_della_rete_con_TMR, TMRReport). `net` non viene modificata.
    Con `layers` vuoto/None restituisce una copia identica e un report vuoto."""
    if merge not in MERGE_MODES:
        raise ValueError(f"merge deve essere uno tra {MERGE_MODES}, non {merge!r}")
    units = resolve_layers(net, layers)
    out = copy.deepcopy(net)
    report = TMRReport(merge=merge)
    for names in units:
        parent_path = _parent_of(names[0])
        parent = out.get_submodule(parent_path) if parent_path else out
        mods = [out.get_submodule(n) for n in names]
        proto = mods[0] if len(mods) == 1 else nn.Sequential(*mods)
        uname = _unit_name(names)
        block = TMRBlock(proto, merge=merge, tmr_name=uname)
        parent._modules[names[0].rsplit('.', 1)[-1]] = block
        for n in names[1:]:                      # il resto del blocco ora vive dentro TMRBlock
            parent._modules[n.rsplit('.', 1)[-1]] = nn.Identity()
        report.units.append(TMRUnit(
            name=uname, modules=names,
            description=' + '.join(_describe(m) for m in mods),
            params_per_replica=sum(p.numel() for m in mods for p in m.parameters())))
    return out, report


# --------------------------------------------------------------------------------------------
# Lato ONNX / FINN (usa solo l'API di ModelWrapper di qonnx: nessun import di FINN)
# --------------------------------------------------------------------------------------------
_TAP_TAG = 'tmr_tap:'


def export_tmr(export_net: nn.Module, args, path: str, export_fn, **export_kwargs):
    """Esporta `export_net` in QONNX con `export_fn` (brevitas.export.export_qonnx).
    Attenzione: con alcuni esportatori di torch (quelli basati su torch.export) l'ottimizzatore fonde le
    repliche identiche e annulla la triplicazione. `check_replicas` lo rileva. In quel caso riesporta con
    export_kwargs={'optimize': False}, che disattiva l'ottimizzatore (puo' lasciare nodi in piu' nel grafo)."""
    return export_fn(export_net, args, path, **export_kwargs)


def check_replicas(model, n_taps: int) -> None:
    """Verifica che nel grafo esportato le repliche siano sopravvissute: n. di uscite atteso e
    ogni uscita prodotta da un nodo diverso e non-Identity (un Identity indica una fusione)."""
    outs = [o.name for o in model.graph.output]
    if len(outs) != 1 + n_taps:
        raise RuntimeError(f'uscite del grafo: {len(outs)}, attese {1 + n_taps}')
    producers = []
    for o in outs:
        p = model.find_producer(o)
        if p is None or p.op_type == 'Identity':
            raise RuntimeError(
                f"l'uscita {o!r} non e' prodotta da una replica distinta (produttore: "
                f"{None if p is None else p.op_type}): l'esportatore ha fuso le repliche identiche. "
                "Riesporta con EXPORT_KWARGS = {'optimize': False}.")
        producers.append(tuple(p.output))      # non id(p): i wrapper protobuf possono essere temporanei
    if len(set(producers)) != len(producers):
        raise RuntimeError('due uscite hanno lo stesso nodo produttore: repliche fuse')


def tag_tap_outputs(model, tap_names: Sequence[str]) -> None:
    """DA CHIAMARE PRIMA di MergeONNXModels, che tiene solo la prima uscita e rinomina tutti i tensori.
    Marca (nel campo doc_string, che sopravvive al merge) il nodo che produce ciascuna uscita extra."""
    extra = list(model.graph.output)[1:]
    if len(extra) != len(tap_names):
        raise RuntimeError(f'{len(extra)} uscite extra nel grafo, {len(tap_names)} nomi attesi')
    for out, name in zip(extra, tap_names):
        model.find_producer(out.name).doc_string = _TAP_TAG + name


def restore_tap_outputs(model):
    """DA CHIAMARE DOPO MergeONNXModels: ricrea le uscite extra dai nodi marcati, con i nomi originali."""
    from onnx import TensorProto, helper
    for node in model.graph.node:
        if node.doc_string.startswith(_TAP_TAG):
            name = node.doc_string[len(_TAP_TAG):]
            node.doc_string = ''
            shape = model.get_tensor_shape(node.output[0])
            model.rename_tensor(node.output[0], name)
            for vi in [v for v in model.graph.value_info if v.name == name]:   # ora e' un'uscita, non un value_info
                model.graph.value_info.remove(vi)
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
    return model


def strip_tap_postprocessing(model, host_ops=('Mul', 'Add', 'Transpose')):
    """DA CHIAMARE DOPO LO STREAMLINING, prima dell'inferenza dei layer HW.
    Le uscite extra escono dal grafo con in coda Transpose (NHWC->NCHW) e Mul/Add (scala, bias): sono
    operazioni da host e non hanno un layer HW, quindi impedirebbero a CreateDataflowPartition di
    chiudere la partizione. Le toglie, cosi' l'uscita extra e' il flusso intero della replica, tale e
    quale esce dall'acceleratore (layout NHWC, scala non applicata)."""
    for out in list(model.graph.output)[1:]:
        while True:
            p = model.find_producer(out.name)
            if p is None or p.op_type not in host_ops:
                break
            dyn = [i for i in p.input if model.get_initializer(i) is None]
            if len(dyn) != 1 or len(model.find_consumers(out.name) or []) > 0:
                break
            new_name = dyn[0]
            shape = model.get_tensor_shape(new_name)
            model.graph.node.remove(p)
            for vi in [v for v in model.graph.value_info if v.name in (new_name, out.name)]:
                model.graph.value_info.remove(vi)
            out.name = new_name
            del out.type.tensor_type.shape.dim[:]
            for d in shape:
                out.type.tensor_type.shape.dim.add().dim_value = int(d)
    return model
