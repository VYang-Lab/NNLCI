<!-- Draft only. Merge into the repository README after review. -->

# NNLCI: Neural Networks with Local Converging Inputs

> A data-centric framework for reconstructing high-fidelity flow fields
> from inexpensive, imperfect, and repeatedly obtainable physics data.

<p align="center">
  <img src="assets/nnlci-lf-to-hf.png"
       alt="Low-fidelity input to high-fidelity target"
       width="100%">
</p>

<p align="center">
  <em>
    Low-fidelity data does not need to be the final answer.<br>
    It only needs to contain enough physics to guide high-fidelity reconstruction.
  </em>
</p>

## Why NNLCI?

- **High-fidelity CFD is expensive beyond runtime.** It requires careful
  meshing, numerical schemes, physical modeling, stabilization, and validation.

- **Low-fidelity physics data is inexpensive and abundant.** It can come from
  automated coarse meshes, simplified CFD, legacy simulations, or sparse experiments.

- **Imperfect does not mean uninformative.** Low-fidelity data still captures
  geometry, boundary conditions, dominant flow structures, and physical trends.

- **NNLCI learns to refine this information.** It transforms low-cost,
  physics-bearing inputs into high-fidelity flow-field predictions.

- **Each HF case carries more training value.** Point-centered local learning
  turns every HF field into many reusable training samples, improving data
  efficiency and generalization across related flow conditions.

- **Predict only where needed.** Each query uses only the local LF neighborhood
  around its point of interest, enabling full-field, regional, boundary, or sparse reconstruction.

- **Scale inference naturally.** Independent point-centered queries can be
  batched or processed in parallel and assembled into the requested output afterward.

- **The investment is reusable.** Once trained for a family of problems, NNLCI
  can rapidly reconstruct new cases without repeating the full HF workflow.

<p align="center">
  <img src="assets/nnlci-concept-overview.png"
       alt="NNLCI Local Converging Inputs Framework Overview"
       width="100%">
</p>


## The NNLCI Philosophy

- **Data first.** The organization and fidelity of the input data matter more
  than building an unnecessarily large neural network.

- **Learn locally.** NNLCI learns reusable local flow relationships instead of
  relying on one monolithic global flow-field mapping.

- **Use every level of physics information.** Numerical and experimental data
  can contribute even when they are coarse, sparse, or imperfect.

- **Fidelity is part of the data structure.** NNLCI uses relationships between
  different fidelity levels as learnable information.

- **Move cost from repetition to learning.** Build representative training data
  once, train the model, and reuse it for rapid reconstruction of new cases.
