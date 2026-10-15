"""Herança de casos (`"$extends"`), remendos mínimos e overrides por caminho.

Um ficheiro de variante traz `"$extends": "base.json"` e só as secções que mudam; `load_json`
carrega a base (recursivamente), funde-lhe a variante com `merge_case` e só depois resolve
`$ref` e valida. Regras da fusão (JSON Merge Patch estendido, sem usar `null` porque `null`
é valor legítimo em `max_step`, `ic.HG`, `pid.*`):

- dicionário sobre dicionário: fusão recursiva, chave a chave; um dicionário do remendo com
  `"$replace": true` substitui o da base por inteiro (é como o remendo apaga chaves);
- lista de objetos com `id` (vértices, arestas) sobre lista do mesmo tipo: fusão por `id`,
  mantendo a ordem da base e acrescentando os ids novos; `{"$replace": [...]}` substitui a
  lista por inteiro;
- qualquer outro valor (escalares, listas sem `id`, como `geometry.points`): substituição.

`make_patch(base, new)` é a inversa: o remendo mínimo tal que `merge_case(base, patch) == new`
(é o que `case_to_json(..., extends=...)` escreve). `apply_overrides` aplica `--set
caminho=valor` com segmentos separados por `.`; um segmento sobre uma lista é o `id` de um
elemento ou, se nenhum id coincidir, um índice (`network.edges.line.D`, `integration.rtol`).
"""

from __future__ import annotations

import copy
import json

REPLACE = "$replace"


def _id_list(x) -> bool:
    return isinstance(x, list) and bool(x) and all(isinstance(e, dict) and "id" in e for e in x)


def merge_case(base, patch):
    """Base com o remendo aplicado (não altera os argumentos)."""
    if isinstance(patch, dict):
        if patch.get(REPLACE) is True:
            return copy.deepcopy({k: v for k, v in patch.items() if k != REPLACE})
        if set(patch) == {REPLACE}:
            return copy.deepcopy(patch[REPLACE])
        if isinstance(base, dict):
            out = {k: copy.deepcopy(v) for k, v in base.items()}
            for k, v in patch.items():
                out[k] = merge_case(base[k], v) if k in base else copy.deepcopy(v)
            return out
        return copy.deepcopy(patch)
    if _id_list(patch) and _id_list(base):
        by_id = {e["id"]: i for i, e in enumerate(base)}
        out = [copy.deepcopy(e) for e in base]
        for e in patch:
            if e["id"] in by_id:
                out[by_id[e["id"]]] = merge_case(base[by_id[e["id"]]], e)
            else:
                out.append(copy.deepcopy(e))
        return out
    return copy.deepcopy(patch)


_SAME = object()  # sentinela: `None` é valor legítimo (null) e não pode significar "sem diferença"


def make_patch(base, new):
    """Remendo mínimo tal que `merge_case(base, patch) == new` (`{}` quando são iguais)."""
    p = _patch(base, new)
    return {} if p is _SAME else p


def _patch(base, new):
    if isinstance(base, dict) and isinstance(new, dict):
        if set(base) - set(new):
            return dict(copy.deepcopy(new), **{REPLACE: True})
        out = {}
        for k, v in new.items():
            if k not in base:
                out[k] = copy.deepcopy(v)
            else:
                p = _patch(base[k], v)
                if p is not _SAME:
                    out[k] = p
        return out or _SAME
    if _id_list(base) and _id_list(new):
        ids_b = [e["id"] for e in base]
        ids_n = [e["id"] for e in new]
        common = [i for i in ids_n if i in ids_b]
        if len(set(ids_b)) != len(ids_b) or len(set(ids_n)) != len(ids_n):
            return copy.deepcopy(new)
        if set(ids_b) - set(ids_n) or common != ids_b or ids_n[: len(common)] != common:
            return {REPLACE: copy.deepcopy(new)}  # ids removidos ou reordenados: lista inteira
        by_id = {e["id"]: e for e in base}
        out = []
        for e in new:
            if e["id"] not in by_id:
                out.append(copy.deepcopy(e))
            else:
                p = _patch(by_id[e["id"]], e)
                if p is not _SAME:
                    out.append(dict({"id": e["id"]}, **p))
        return out or _SAME
    return _SAME if base == new else copy.deepcopy(new)


def _step(obj, seg, create=False):
    if isinstance(obj, list):  # primeiro por `id` (os ids legados são dígitos), depois por índice
        for i, e in enumerate(obj):
            if isinstance(e, dict) and str(e.get("id")) == seg:
                return i
        if seg.lstrip("-").isdigit() and -len(obj) <= int(seg) < len(obj):
            return int(seg)
        raise KeyError(f"nenhum elemento com id ou índice {seg!r}")
    if seg not in obj:
        if not create:
            raise KeyError(seg)
        obj[seg] = {}
    return seg


def parse_override(text: str):
    """`caminho=valor` -> (caminho, valor); o valor é JSON quando possível, senão texto."""
    path, sep, value = text.partition("=")
    if not sep or not path:
        raise ValueError(f"override espera caminho=valor: {text!r}")
    try:
        return path.strip(), json.loads(value)
    except json.JSONDecodeError:
        return path.strip(), value


def apply_overrides(d: dict, overrides) -> dict:
    """Aplica overrides (`{"integration.rtol": 1e-3}` ou lista de `caminho=valor`) ao dicionário
    do caso, no lugar; chaves inexistentes no fim do caminho são criadas."""
    items = overrides.items() if isinstance(overrides, dict) else [parse_override(t) for t in overrides]
    for path, value in items:
        segs = path.split(".")
        obj = d
        for seg in segs[:-1]:
            obj = obj[_step(obj, seg, create=True)]
        obj[_step(obj, segs[-1], create=True)] = value
    return d
