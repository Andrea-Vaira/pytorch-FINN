"""Test di tmr_layers.py. Esegui con:  python test_tmr_layers.py   (oppure: pytest test_tmr_layers.py)"""
import copy

import torch
import torch.nn as nn

from tmr_layers import (TMRBlock, TMRExportWrapper, apply_tmr, logical_layers, tmr_blocks)

torch.manual_seed(0)


def make_net():
    """Stessa struttura di SmallCNN, con layer torch standard (i test non richiedono Brevitas)."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 16, 3, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2))
            self.classifier = nn.Sequential(
                nn.Linear(800, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Linear(64, 10, bias=False))

        def forward(self, x):
            return self.classifier(self.features(x).flatten(1))

    net = Net()
    for m in net.modules():              # statistiche BN non banali
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.running_mean.normal_(0, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    return net.eval()


X = torch.randn(4, 1, 28, 28)


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f'attesa {exc.__name__}')


def test_logical_layers():
    names = [b.names for b in logical_layers(make_net())]
    assert names == [('features.0', 'features.1', 'features.2'),
                     ('features.3', 'features.4', 'features.5'),
                     ('features.7', 'features.8', 'features.9'),
                     ('classifier.0', 'classifier.1', 'classifier.2'),
                     ('classifier.3',)], names


def test_original_untouched_and_replicas_independent():
    net = make_net()
    tnet, rep = apply_tmr(net, ['features.3'], merge='median')
    assert not tmr_blocks(net) and len(tmr_blocks(tnet)) == 1
    blk = tmr_blocks(tnet)[0][1]
    ptrs = [r.weight.data_ptr() for r in blk.replicas]
    assert len(set(ptrs)) == 3 and net.features[3].weight.data_ptr() not in ptrs
    assert rep.tag == 'tmr-median__features_3' and rep.units[0].params_per_replica == 16 * 16 * 9


def test_equivalence_all_specs_and_modes():
    net = make_net()
    ref = net(X)
    for merge in ('tap', 'median'):
        for spec in (1, 'features.3', [1, 3], [('features.3', 'features.4', 'features.5')],
                     [0, 1, 2, 3, 4], ['features.0', 'classifier.3']):
            tnet, _ = apply_tmr(net, spec, merge=merge)
            assert torch.equal(tnet.eval()(X), ref), (merge, spec)


def test_block_spec_replaces_rest_with_identity():
    tnet, rep = apply_tmr(make_net(), [('features.3', 'features.4', 'features.5')])
    assert isinstance(tnet.features[3], TMRBlock)
    assert isinstance(tnet.features[4], nn.Identity) and isinstance(tnet.features[5], nn.Identity)
    assert rep.units[0].name == 'features_3-5'


def test_median_masks_one_faulty_replica_but_not_two():
    net = make_net()
    ref = net(X)
    tnet, _ = apply_tmr(net, 'features.3', merge='median')
    blk = tmr_blocks(tnet)[0][1]
    with torch.no_grad():
        blk.replicas[1].weight.mul_(-5.0)                       # guasto su una replica
    assert torch.equal(tnet(X), ref)
    with torch.no_grad():
        blk.replicas[2].weight.mul_(-5.0)                       # guasto identico su due: il voto fallisce
    assert not torch.equal(tnet(X), ref)


def test_tap_forward_and_export_wrapper():
    net = make_net()
    ref = net(X)
    tnet, _ = apply_tmr(net, [1, 3], merge='tap')
    assert torch.equal(tnet(X), ref)                           # forward normale: solo l'uscita principale
    w = TMRExportWrapper(tnet)
    outs = w(X)
    assert isinstance(outs, tuple) and len(outs) == 1 + 2 * 2
    assert w.tap_names == ['features_3-5_r1', 'features_3-5_r2', 'classifier_0-2_r1', 'classifier_0-2_r2']
    assert torch.equal(outs[0], ref)
    assert all(b._collect is None for _, b in tmr_blocks(tnet))   # nessuno stato residuo
    # senza blocchi tap l'export restituisce un tensore singolo
    assert isinstance(TMRExportWrapper(apply_tmr(net, [1], merge='median')[0])(X), torch.Tensor)
    assert isinstance(TMRExportWrapper(apply_tmr(net, [])[0])(X), torch.Tensor)


def test_tap_outputs_equal_when_healthy_and_differ_when_faulty():
    net = make_net()
    tnet, _ = apply_tmr(net, 'features.3', merge='tap')
    w = TMRExportWrapper(tnet)
    main, t1, t2 = w(X)
    ref_block = net.features[3](net.features[2](net.features[1](net.features[0](X))))
    assert torch.equal(t1, ref_block) and torch.equal(t2, ref_block)
    with torch.no_grad():
        tmr_blocks(tnet)[0][1].replicas[2].weight.zero_()
    _, t1b, t2b = w(X)
    assert torch.equal(t1b, ref_block) and not torch.equal(t2b, ref_block)


def test_empty_selection():
    net = make_net()
    for empty in (None, [], ()):
        tnet, rep = apply_tmr(net, empty)
        assert not rep.enabled and rep.tag == 'baseline' and not tmr_blocks(tnet)
        assert torch.equal(tnet(X), net(X))


def test_errors():
    net = make_net()
    raises(IndexError, lambda: apply_tmr(net, 9))
    raises(KeyError, lambda: apply_tmr(net, 'features.99'))
    raises(KeyError, lambda: apply_tmr(net, 'nonesiste'))
    raises(ValueError, lambda: apply_tmr(net, ['features.3', 'features.3']))
    raises(ValueError, lambda: apply_tmr(net, [1, ('features.3', 'features.4')]))       # sovrapposti
    raises(ValueError, lambda: apply_tmr(net, [('features.3', 'features.5')]))          # non consecutivi
    raises(ValueError, lambda: apply_tmr(net, [('features.3', 'classifier.0')]))        # contenitori diversi
    raises(ValueError, lambda: apply_tmr(net, 1, merge='boh'))
    raises(TypeError, lambda: apply_tmr(net, [1.5]))


def test_double_application_is_rejected():
    tnet, _ = apply_tmr(make_net(), 'features.3')
    raises(ValueError, lambda: apply_tmr(tnet, 'features.3'))                  # gia' un TMRBlock
    raises(ValueError, lambda: apply_tmr(tnet, 'features.3.replicas.0'))       # dentro un TMRBlock
    assert len(tmr_blocks(tnet)) == 1


def test_brevitas_if_available():
    try:
        from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU
    except Exception:
        print('  (brevitas non installato: test saltato)')
        return

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                QuantConv2d(1, 16, 3, bias=False, weight_bit_width=4), nn.BatchNorm2d(16), QuantReLU(bit_width=4),
                QuantConv2d(16, 16, 3, bias=False, weight_bit_width=4), nn.BatchNorm2d(16), QuantReLU(bit_width=4),
                nn.MaxPool2d(2),
                QuantConv2d(16, 32, 3, bias=False, weight_bit_width=4), nn.BatchNorm2d(32), QuantReLU(bit_width=4),
                nn.MaxPool2d(2))
            self.classifier = nn.Sequential(
                QuantLinear(800, 64, bias=False, weight_bit_width=4), nn.BatchNorm1d(64), QuantReLU(bit_width=4),
                QuantLinear(64, 10, bias=False, weight_bit_width=4))

        def forward(self, x):
            return self.classifier(self.features(x).flatten(1))

    net = Q().eval()
    assert [b.names for b in logical_layers(net)] == [b.names for b in logical_layers(make_net())]
    ref = net(X)
    for merge in ('tap', 'median'):
        tnet, _ = apply_tmr(net, [1, 4], merge=merge)
        assert torch.equal(tnet.eval()(X), ref), merge
    # i quantizzatori Brevitas non devono comparire come layer a se'
    assert all('quant' not in n for b in logical_layers(net) for n in b.names)


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    for n, f in tests:
        f()
        print('OK  ', n)
    print(f'\n{len(tests)} test passati')
