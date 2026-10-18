# Licensing

This fork preserves the upstream licenses. It does not apply a new blanket
license to the upstream research repository or its datasets.

| Scope | License |
| --- | --- |
| Upstream code under `metric/` | [MIT, RoSE Developers](metric/LICENSE.txt) |
| Bundled Smatch++ under `experiment/esmatchpp/` and our changes to it | [GPLv3](experiment/esmatchpp/LICENSE) |
| New `metric/amr_normalization.py` and our changes to the RoSE code | [MIT, Mirza Husadzic](LICENSES/fork-MIT.txt) |
| New root packaging, documentation, tests and example | [GPL-3.0-only](experiment/esmatchpp/LICENSE), copyright 2026 Mirza Husadzic |

The `semcat-runtime` distribution includes both MIT and GPLv3 code; its SPDX
expression is `GPL-3.0-only AND MIT`. It must not be presented as an MIT-only
package. Distribution of the combined runtime must comply with GPLv3, including
its corresponding-source requirements, and retain the MIT notices.

Upstream's Smatch++ metadata identifies Juri Opitz as its author and
<https://github.com/flipz357/smatchpp> as its source. The SEMCAT modifications
come from <https://github.com/snucclab/ssharp>. The original license texts and
copyright notices are retained verbatim, including upstream placeholders.

The wheel and source distribution include these notices and license texts.
They package only the runtime, its resources and supporting files; they do not
include the repository's experimental datasets. No additional rights to those
datasets or other upstream material are asserted here.

See [CHANGELOG.md](CHANGELOG.md) and the dated notices in changed source files
for this fork's modifications. Integrating the runtime into another application
requires considering that application's distribution and license; calling it
through a subprocess does not by itself settle that question.
