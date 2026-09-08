# NYC-STEPS experimental splits

Source: [Massive-STEPS-New-York](https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-New-York), released by CRUISE Research Group (UNSW). The source dataset card declares Apache-2.0. A copy is provided in LICENSE.txt. Original dataset attribution remains with its creators; these files are a processed derivative, not a new original dataset.

The local conversion scripts parse trajectory inputs and targets, remap user/POI identifiers, and export RecBole tab-separated sequences. These files are copied from the local experimental data directory without regenerating or changing the splits. The exact correspondence to each historical run has not been independently reverified.

Included: train/validation/test interaction files; original-to-local user and item ID maps; feature-row mapping. The interaction fields are user_id:token, item_id_list:token_seq, and item_id:token.

Feature matrices and Faiss indices are not included. The source-conversion tools are under tools/; their hashed features are not pretrained language-model embeddings. Keep existing splits fixed when reconstructing supplementary assets.

This license applies to this derived data release, not automatically to all repository code. Consult the original dataset card for citation details and provenance.
