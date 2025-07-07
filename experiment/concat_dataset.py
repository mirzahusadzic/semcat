from argparse import ArgumentParser
from pathlib import Path

from penman import load as penman_load, dump as penman_dump
from penman.models.amr import model as AMR_MODEL
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', '-in', '-i', type=Path, nargs='+',
                        help='List of input AMR annotation files')
    parser.add_argument('--output', '-out', '-o', type=Path,
                        help='Output file for concatenated AMRs')
    args = parser.parse_args()

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    penman_dump([amr
                 for file in args.input
                 for amr in tqdm(penman_load(file, model=AMR_MODEL), desc=f'{file}')],
                file=args.output, indent=2)
