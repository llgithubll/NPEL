# NPEL
**NPEL**(Neural Paired Entity Linking) is the entity linking method proposed by "Neural Paired Entity Linking in Web Tables"
 
### Datasets

There are three datasets used in NPEL experiment. `Web_Manual_fixed_2020`(with 347 tables), `Wiki_Links_Large`(with 168 tables),`Wiki_Links_Random_2020`(with 2335 tables).

Director `Web_Manual_Fixed_2020` has two subdirectors. One of them is table data in json format and another director is the labeled entity correspondent to table data in the same position.

Director `Wiki_Links_Large` and `Wiki_Links_Random_2020` share the same struct of files. They have only one `tables` director. All table data and labeled are in json type files. Each table mention labeled entity is annotate as `target` in `surfaceLinks`

### Code

`code` contains the mention-entity semantics model and pair-linking algorithm implementation.

The mention-entity semantics model mainly included in `nn_models` and pair-linking algorithm is implemented in `pair_linking.py`

Both neural network model and pair-linking algorithm required input data format each mention in a table contains all it's embedding, candidates, candidates' embedding, candidates' abstract. These elements are directly accessed as dict in model or algorithm.

Besides, we utilized [this work](https://github.com/dalab/deep-ed/blob/master/data_gen/gen_p_e_m/gen_p_e_m_from_wiki.lua) to generate prior data.

### run-step




