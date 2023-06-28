# Supporting-data-for-Data-driven-discovery-of-innate-immunomodulators-

## File Structure

- bo/: Bayesian optimization (BO) module, including the Gaussian Process Regreassion (GPR) surrogate model.

- data/:

    - screening_library.csv: The information of the compounds in the design space (searching pool) of this project that were virtually screened.

    - screening_results.csv: The information of the compounds in this project that were experimentally assayed and screened.

    - vae_training_library.txt: The SMILES representation of the compound library used to train the VAE model.

- plot/: Jupyter notebooks containing codes to generate the plots and figures.

- vae/: Variational autoencoder (VAE) module. A neural network consisting of 2 parts - encoders and decoders - which converts compound SMILES/SELFIES representation into continuous numeric embeddings and converts back to SMILES/SELFIES representation. 