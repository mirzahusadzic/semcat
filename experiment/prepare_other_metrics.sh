#!/bin/bash
# File for downloading S2MATCH, SEMBLEU, and WWLK series.

# Download SEMBLEU
if [[ ! -d sembleu ]]
then
    git clone https://github.com/freesunshine0316/sembleu.git
    cd sembleu
    git checkout 69ff2d496311eaefdaeb9d5cb4fcf2309a7a12b4

    chmod +x eval.sh
    cp ../sembleu_sentwise_eval.py src/per_sentence_eval.py
    cd ..
fi

# Download S2MATCH
if [[ ! -d amr-metric-suite ]]
then
    git clone https://github.com/flipz357/amr-metric-suite.git
    cd amr-metric-suite
    git checkout 9413552813cd3cfd4b709ed23c1aa847d9576970

    ./download_glove.sh
    cd ..
fi

# Download WWLK series
if [[ ! -d weisfeiler-leman-amr-metrics ]];
then
    git clone https://github.com/flipz357/weisfeiler-leman-amr-metrics.git
    cd weisfeiler-leman-amr-metrics
    git checkout b351fbd99ae394b79753d4d6b420613308db69e5

    # Download data for WWLK-theta
    git clone https://github.com/flipz357/bamboo-amr-benchmark.git
    cd bamboo-amr-benchmark
    git checkout 3152f704bdad29ddc48c51314bcd271362196755
    cd ../..
fi

# Download SEMA
if [[ ! -d sema ]];
then
    git clone https://github.com/rafaelanchieta/sema
    cd sema
    git checkout 9a4911cb76c0dd6329b52d518f584bb56e1c2cf7
    cp ../sema_per_eval.py ./
    cd ..
fi

# Download SMATCH++
if [[ ! -d smatchpp ]];
then
    git clone https://github.com/flipz357/smatchpp.git
    cd smatchpp
    git checkout 7ba752c7bb65f448f60d24cef4c4c426ea54cad2
    cp -f ../smatchpp_bindings_patch.py smatchpp/bindings.py
    cd ..
fi

# Install requirements
pip install networkx scipy numpy penman gensim pyemd mip
