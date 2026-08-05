# Upload package manifest

This package reorganizes the validated NNLCI workflows into four problem
directories. It intentionally excludes Python environments, IDE caches,
Visual Studio build intermediates, smoke-test weights, and duplicate root-level
legacy files.

Included material:

- runnable Python, C++, and CUDA Fortran source;
- supplied and regenerated CFD/training data;
- non-smoke trained model weights;
- prediction arrays, metrics, figures, and publication plots;
- Config 3 bootstrap and processed 2D arrays;
- documentation assets and legacy documentation.

The 2D Config 3 model in this package is the two-epoch pipeline-validation
checkpoint generated from bootstrap data. It is not a publication-quality
scientific model. The supplied 1D `best_model_*.pth` files are the original
trained checkpoints used for the reported prediction metrics.

The package includes files larger than GitHub's regular 100 MB limit. The root
`.gitattributes` therefore routes arrays, weights, CFD fields, and PDFs through
Git LFS. Run `git lfs install` before the first commit to a GitHub repository.
