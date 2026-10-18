# Changes from upstream

## 0.1.0 — 2026-10-19

Based on `snucclab/ssharp` at
`5c5d4bd5a15cf1e1a0db05b16610ee39a90f7c2d`. Upstream history is preserved.

The initial four changes were extracted from a consumer's verified patch set:

1. **Local normalization:** canonicalize AMR roles before removing inverses,
   deduplicate after normalization and optionally return original variable IDs.
2. **Global normalization:** use the same role-normalization function in RoSE.
3. **Normalization origins:** expose each pair's normalized PENMAN and surviving
   original IDs through `Smatchpp.last_normalization`.
4. **Lossless labels:** encode RoSE concept and attribute labels with ASCII JSON
   escapes instead of transliteration, so `Malmo` and `Malmö` remain distinct.

Standalone regression testing identified two additional corrections:

- Use the AMR role model in Smatch++'s later deinversion step, which previously
  reversed every role containing `-of`, including lexical `:consist-of` and
  `:prep-out-of`.
- Use PENMAN's AMR decoder in Smatch++'s reader to preserve quoted whitespace,
  escapes and parentheses. The old token parser collapsed `"New York"` into
  `"NewYork"`. Preserve the reader's existing synthetic root triple.

Add root packaging, an executable example, installed-package tests and CI.
Remove RoSE's now-unused `anyascii` import and dependency. Preserve the optimizer,
global matching algorithm and upstream aggregation implementation.

### Scoring profile

The shared role profile is `amr-normalized-roles-v1`. It uses PENMAN's AMR aliases
and inverse rules, then represents `:domain` as inverse `:mod`. That last step is
an explicit profile choice: PENMAN itself treats both as canonical roles.
Lexical `:consist-of`, `:prep-out-of` and `:prep-on-behalf-of` keep their direction;
genuine inverses such as `:ARG0-of` reverse. Existing dereification, duplicate
removal and canonical rerooting remain in place.

`to_standard_amr(text)` still returns a string. With `with_origins=True` it returns
`(text, origins)`, where keys are normalized variables and values are original
variables. Dereified-away nodes have no entry. `Smatchpp.last_normalization`
describes the two graphs after canonicalization and before the configurable
Smatch++ preprocessor. To relate solver IDs to these origins, disable further
variable relabeling (`relabel_vars=False`); see the tests. These are surviving
variable identities, not a complete semantic provenance map.

The upstream Smatch++ defaults still lowercase text and strip quotes. Applications
that need literal distinctions should set `lower=False, remove_quote=False`,
as the example does. RoSE ignores `:wiki` by default; apply the same wiki policy
to both graphs before local scoring when combining metrics.

Upstream `build_semcat` normalizes raw weights **1** and **1 − alpha**:

```text
SEMCAT = (local + (1 - alpha) * global) / (2 - alpha)
```

At `alpha=0.25`, this is `(local + 0.75 * global) / 1.75`. The fork does not
replace it with a conventional alpha blend. Record package, dependency and
profile versions with results; corrected normalization changes some scores.
