#!/usr/bin/env python3
"""Build the traced substrate G_traced from the MaleCNS v1.0 flat tables and assign presynaptic signs.

Reproduces the ballot's headline figures (ompi, grant 1fab0 proposal 21): 165,122 traced bodies,
25,563,197 edges, 124,025,046 synapse weight; flat table 151,856,684 rows / 311,833,243 weight.
Outputs (data/derived/, not committed): G_traced_post_by_pre.npz (CSR, rows = postsynaptic body index,
cols = presynaptic body index, values = synapse count), G_traced_bodyIds.npy (index -> bodyId),
G_traced_presyn_sign.npy (+1 / -1 / 0 per presynaptic body), G_traced_signed.npz (signed CSR, zero-sign
presynaptic edges dropped), and battery/substrate-figures.json + battery/sign-rule.json (committed).
Sign rule: consensus_nt, falling back to predicted_nt when consensus is 'unclear' or missing;
acetylcholine +1; gaba, glutamate, histamine -1; dopamine, octopamine, serotonin, unclear, missing 0.
Shiu et al. 2024 treat the monoamines as excitatory; we drop them and say so. Run: .venv/bin/python src/prep_substrate.py
"""
import json, sys, time, collections, os
import numpy as np, scipy.sparse as sp
import pyarrow.feather as f, pyarrow.compute as pc
D = os.environ.get('FLY_DATA', 'data/malecns'); OUT = os.environ.get('FLY_DERIVED', 'data/derived'); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
ann = f.read_table(f'{D}/body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['bodyId', 'status'])
ids = np.sort(np.array(ann.filter(pc.equal(ann.column('status'), 'Traced')).column('bodyId').to_pylist(), dtype=np.int64))
w = f.read_table(f'{D}/connectome-weights-male-cns-v1.0-minconf-0.5.feather')
pre = w.column('body_pre').to_numpy(); post = w.column('body_post').to_numpy(); wt = w.column('weight').to_numpy()
def index_of(x):
    p = np.searchsorted(ids, x); ok = (p < len(ids)) & (ids[np.minimum(p, len(ids) - 1)] == x); return p, ok
pi, okp = index_of(pre); qi, okq = index_of(post); m = okp & okq
n = len(ids)
A = sp.csr_matrix((wt[m].astype(np.float32), (qi[m], pi[m])), shape=(n, n))
import hashlib
def sha256_of(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(bufsize), b''): h.update(chunk)
    return h.hexdigest()
CANONICAL_WEIGHTS_SHA256 = 'e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1'   # ompi, grant 1fab0 proposal 21
input_hashes = {os.path.basename(p): sha256_of(p) for p in (f'{D}/connectome-weights-male-cns-v1.0-minconf-0.5.feather', f'{D}/body-annotations-male-cns-v1.0-minconf-0.5.feather', f'{D}/body-neurotransmitters-male-cns-v1.0.feather')}
weights_ok = input_hashes['connectome-weights-male-cns-v1.0-minconf-0.5.feather'] == CANONICAL_WEIGHTS_SHA256
print('input sha256:', input_hashes, 'weights canonical:', weights_ok)
figures = {'traced_bodies': int(n), 'edges': int(A.nnz), 'weight': int(wt[m].sum()), 'flat_rows': int(len(pre)), 'flat_weight': int(wt.sum()), 'input_sha256': input_hashes, 'weights_match_canonical': weights_ok, 'restriction_predicate': 'status == Traced on both endpoints (Traced x Traced induced subgraph of the minconf-0.5 flat table)'}
expected = {'traced_bodies': 165122, 'edges': 25563197, 'weight': 124025046, 'flat_rows': 151856684, 'flat_weight': 311833243}
print('substrate figures:', {k: figures[k] for k in expected}, 'MATCH' if {k: figures[k] for k in expected} == expected else 'MISMATCH vs ballot ' + str(expected))
sp.save_npz(f'{OUT}/G_traced_post_by_pre.npz', A); np.save(f'{OUT}/G_traced_bodyIds.npy', ids)
nt = f.read_table(f'{D}/body-neurotransmitters-male-cns-v1.0.feather', columns=['body', 'consensus_nt', 'predicted_nt']).to_pandas().set_index('body')
cons = nt['consensus_nt'].reindex(ids).fillna('missing').to_numpy(); pred = nt['predicted_nt'].reindex(ids).fillna('missing').to_numpy()
lab = np.where(np.isin(cons, ['unclear', 'missing']), pred, cons)
# Grant 1FAB0 Dare (Vish #7586, Quire c91300): Shiu et al. 2024 monoamines rule
# Setting dopamine, octopamine, and serotonin to +1 restores 903 bodies and 637,149 edges.
# Supported via --monoamines flag or FLY_MONOAMINES_EXCITATORY=1 env var.
use_mono = ('--monoamines' in sys.argv or '--monoamines-excitatory' in sys.argv or os.environ.get('FLY_MONOAMINES_EXCITATORY') == '1')
if use_mono:
    sign_map = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1, 'histamine': -1, 'dopamine': 1, 'octopamine': 1, 'serotonin': 1}
    sign_rule_str = "Shiu et al. 2024 monoamines rule: consensus_nt, falling back to predicted_nt when consensus is 'unclear' or missing; acetylcholine, dopamine, octopamine, serotonin +1; gaba, glutamate, histamine -1; unclear, missing 0."
else:
    sign_map = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1, 'histamine': -1}
    sign_rule_str = __doc__.split('Sign rule: ')[1].split('\n')[0] + ' ' + __doc__.split('Sign rule: ')[1].split('\n')[1].strip()
sign = np.array([sign_map.get(l, 0) for l in lab], dtype=np.int8)
np.save(f'{OUT}/G_traced_presyn_sign.npy', sign)
S = A.tocsc().multiply(sign[None, :].astype(np.float32)).tocsr(); S.eliminate_zeros(); sp.save_npz(f'{OUT}/G_traced_signed.npz', S)
rule = {'sign_rule': sign_rule_str,
        'labels': dict(collections.Counter(lab.tolist())), 'excitatory': int((sign > 0).sum()), 'inhibitory': int((sign < 0).sum()), 'zero': int((sign == 0).sum()),
        'signed_nnz': int(S.nnz), 'dropped_edges_from_zero_sign_presyn': int(A.nnz - S.nnz)}
rule_out = 'battery/sign-rule-monoamines.json' if use_mono else 'battery/sign-rule.json'
json.dump(figures, open('battery/substrate-figures.json', 'w'), indent=1); json.dump(rule, open(rule_out, 'w'), indent=1)
print('signs:', {k: rule[k] for k in ('excitatory', 'inhibitory', 'zero', 'signed_nnz', 'dropped_edges_from_zero_sign_presyn')}, '%.0fs' % (time.time() - t0))

