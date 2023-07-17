# A Bilevel Formalism for the Peer-Reviewing Problem

This is the official code of our paper which has been accepted by the European Conference on Artificial Intelligence ([ECAI](https://ecai2023.eu/)) 2023.

A Bilevel Formalism for the Peer-Reviewing Problem

by Gennaro Auricchio, Ruixiao Zhang, [Jie Zhang](https://researchportal.bath.ac.uk/en/persons/jie-zhang), [Xiaohao Cai](https://www.southampton.ac.uk/people/5y65yy/doctor-xiaohao-cai)

## Dependencies

- [Julia 1.8.5](https://julialang.org/downloads/)

### Packages

Packages need to be [imported into the Julia environment](https://doc.cocalc.com/howto/install-julia-package.html) before running the scripts. All used packages are stated at the beginning of the scripts.

### Dataset

We use the [Data Set for Multi-Aspect Review Assignment Evaluation](https://timan.cs.illinois.edu/ir/data/ReviewData.zip) from the University of Illinois Urbana-Champaign ([UIUC](https://illinois.edu/index.html)). 

## Get started

Run relevant .jl files to reproduce the results in the paper. 

for example,

loop-best-u6-latest.jl -> Aligned case with U=6.

loop-random-uni-u8-latest.jl -> Random case with U=8 and using the uniform distribution for the random matrix.


