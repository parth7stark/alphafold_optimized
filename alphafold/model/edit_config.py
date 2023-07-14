from pathlib import Path
from functools import wraps, partialmethod
from typing import Tuple, List, Optional
import haiku
from alphafold.model import model, data
from alphafold.model.modules import AlphaFold
from alphafold.model.modules_multimer import AlphaFold as AlphaFoldMultimer
from ml_collections.config_dict import ConfigDict

def load_models_config(
    config: ConfigDict = None,
    num_recycles: Optional[int] = None,
    recycle_early_stop_tolerance: Optional[float] = None,
    num_ensemble: int = 1,
    model_order: Optional[List[int]] = None,
    max_seq: Optional[int] = None,
    max_extra_seq: Optional[int] = None,
    use_fuse: bool = True,
    use_bfloat16: bool = True,
    use_dropout: bool = False,
    save_all: bool = True,
    model_suffix: str = "multimer"
) -> List[Tuple[str, model.RunModel, haiku.Params]]:
    """We use only two actual models and swap the parameters to avoid recompiling.

    Note that models 1 and 2 have a different number of parameters compared to models 3, 4 and 5,
    so we load model 1 and model 3.
    """

    if model_order is None:
        model_order = [1, 2, 3, 4, 5]
    else:
        model_order.sort()

    # model_build_order = [3, 4, 5, 1, 2]
    # if "multimer" in model_suffix:
    #     models_need_compilation = [3]
    # else:
    #     # only models 1,2 use templates
    #     models_need_compilation = [1, 3] if use_templates else [3]
    
    model_runner = None
    for model_number in model_build_order:
        # if model_number in models_need_compilation:

        # get configurations
        # model_config = config.model_config("model_" + str(model_number) + model_suffix)
        model_config = config #from run_alphafold.py
        # model_config.model.stop_at_score = float(stop_at_score)
        # model_config.model.rank_by = rank_by

        # set dropouts
        model_config.model.global_config.eval_dropout = use_dropout

        # set bfloat options
        model_config.model.global_config.bfloat16 = use_bfloat16
        
        # set fuse options
        model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights = use_fuse
        model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights = use_fuse
        
        # if "multimer" in model_suffix or model_number in [1,2]:
        model_config.model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_incoming.fuse_projection_weights = use_fuse
        model_config.model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_outgoing.fuse_projection_weights = use_fuse
                    
        # set number of sequences options
        if max_seq is not None:
            if "multimer" in model_suffix:
                model_config.model.embeddings_and_evoformer.num_msa = max_seq
            else:
                model_config.data.eval.max_msa_clusters = max_seq
        
        if max_extra_seq is not None:
            if "multimer" in model_suffix:
                model_config.model.embeddings_and_evoformer.num_extra_msa = max_extra_seq
            else:
                model_config.data.common.max_extra_msa = max_extra_seq

        # disable some outputs if not being saved
        if not save_all:
            model_config.model.heads.distogram.weight = 0.0
            model_config.model.heads.masked_msa.weight = 0.0
            model_config.model.heads.experimentally_resolved.weight = 0.0

        # set number of recycles and ensembles            
        if "multimer" in model_suffix:
            if num_recycles is not None:
                model_config.model.num_recycle = num_recycles
            # model_config.model.embeddings_and_evoformer.use_cluster_profile = use_cluster_profile
            model_config.model.num_ensemble_eval = num_ensemble
        else:
            if num_recycles is not None:
                model_config.data.common.num_recycle = num_recycles
                model_config.model.num_recycle = num_recycles
            model_config.data.eval.num_ensemble = num_ensemble


        if recycle_early_stop_tolerance is not None:
            model_config.model.recycle_early_stop_tolerance = recycle_early_stop_tolerance
            
    return model_config
