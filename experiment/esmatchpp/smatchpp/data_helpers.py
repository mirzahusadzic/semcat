# Modified 2026-10-19 by Mirza Husadzic: lossless PENMAN AMR decoding.
# SPDX-License-Identifier: GPL-3.0-only; see ../LICENSE.
import logging
import penman
from penman.models.amr import model as amr_model
from smatchpp import interfaces

logger = logging.getLogger("__main__")


def read_amr_strings_from_file(filepath):

    with open(filepath, "r") as f:
        amrs_meta = f.read().split("\n\n")

    amrs = ["\n".join([l for l in am.split("\n") if not l.startswith("# ::")]) for am in amrs_meta]
    if not amrs[-1]:
        logger.debug("removing last line which is empty")
        amrs = amrs[:-1]
    return amrs


class PenmanReader(interfaces.GraphReader):

    def __init__(self):
        return None

    def _string2graph(self, string):
        """Decode AMR without changing quoted text or lexical role direction."""
        graph = penman.decode(string, model=amr_model)
        return list(graph.triples) + [("ROOT_OF_GRAPH", ":root", graph.top)]


class TSVReader(interfaces.GraphReader):

    def _string2graph(self, string):

        triples = string.split("\n")
        triples = [tuple(triple.split()) for triple in triples]
        triples = [(triple[0], triple[2], triple[1]) for triple in triples]

        return triples
    

def get_reader(reader_name):

    if reader_name == "penman":
        return PenmanReader()

    if reader_name == "tsv":
        return TSVReader()
    
    raise NameError("reader name not known, available: penman, tsv")

