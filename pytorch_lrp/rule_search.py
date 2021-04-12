import os, sys
os.environ["NCCL_DEBUG"]="INFO"
sys.path.append('/project/ppi_workspace/toil-env36/lib/python3.6/site-packages')
sys.path.append("/home/ed4bu/BSP-LRP")

import uuid
from itertools import product

from torch import nn
from scorer import ProteinScorer, gmean
from Models2 import AutoencoderModel

from pytorch_lrp import rules
rule_names = {r.__class__.__name__.split(".")[-1]:r for r in rules}

def setup_lrp(job, rule_name, **rule_attrs):
    for pdb_file in os.listdir("/project/ppi_workspace/TestUrfoldScorer"):
        if pdb_file.endswith(".pdb") and os.isfile(pdb_file.rsplit(".", 1)[0]+"_atom.h5"):
            job.addChildJobFn(lrp, pdb_file, rule, **rule_attrs)

def lrp(job, pdb_file, rule_name, **rule_attrs):
    rule = rule_names[rule_name]
    for attr, val in rule_attrs.iteritems():
        if attr.endswith("RULE"):
            val = rule_names[val]
        setattr(rule, attr, value)

    use_features = [
         'C', 'CA', 'N', 'O', 'OH', 'C_elem', 'N_elem', 'O_elem', 'S_elem',
          'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
          'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
          'is_helix', 'is_sheet', 'residue_buried', 'residue_exposed','atom_is_buried',
          'atom_exposed',  'is_hydrophbic', 'pos_charge',  'is_electronegative'
    ]

    cath_domain = os.path.basename(pdb_file.rsplit(".", 1)[0])

    criterion = nn.MSELoss()
    scorer = ProteinScorer.predict(
        AutoencoderModel, #Model Class
        criterion, #Loss Function
        pdb_file, #Path to pdb file
        cath_domain, #CATH domain name
        features_path="/project/ppi_workspace/TestUrfoldScorer",
        use_features=use_features,
        autoencoder=True,
        weight_files=["/project/ppi_workspace/TestUrfoldScorer/skipconn_false_ig.pth"],
        model_kwds={"nIn":len(use_features), "skipconnections": False},
        lrp_rule=rule}
    )

    base_dir = os.getcwd()

    agg_fucs = [np.max, np.min, np.sum, np.mean, gmean]
    for i, (voxel_agg, feature_agg, residue_agg) in enumerate(product(agg_funcs, repeat=3)):
        structure = scorer.structure.copy()
        structure.atom_features = pd.DataFrame(np.nan,
            index=structure.atom_features.index,
            columns=scorer.output_labels)

        scorer.LRP_merge2(scorer.relevance, structure, voxel_agg=voxel_agg,
            feature_agg=feature_agg, residue_agg=residue_agg)

        unique_dirname = os.path.join(base_dir, str(uuid.uuid4()))
        os.path.makedirs(unique_dirname)
        os.chdir(unique_dirname)

        structure.write_features_to_pdb(features=["total_residue_relevance"], name="{}-lrp".format(i))
        structure.write_features_to_pdb(features=["total_relevance"], name="{}-lrp".format(i))

        with open("LRP_run_{}-{}.txt".format(cath_domain, i)) as f:
            print("LOSS", scorer.loss, file=f)
            print("RULE:", rule, file=f)
            print("VOXEL AGG:", voxel_agg, file=f)
            print("FEATURE AGG:", feature_agg, file=f)
            print("RESIDUE AGG:", residue_agg, file=f)

    os.chdir(base_dir)

def start_toil(job):
    for rule in ("GRule", "GRuleRelu", "ERule", "ERuleRelu", "ZeroRule",
                 "ZeroRuleRelu", "Alpha1Beta0", "Alpha2Beta1"):
        job.addChildJobFn(setup_lrp, rule)

    for upper_rule in ("GRule", "GRuleRelu"):
        for middle_rule in ("ERule", "ERuleRelu"):
            for lower_rule in ("ZeroRule", "ZeroRuleRelu", "Alpha1Beta0", "Alpha2Beta1"):
                for lower_split in range(0, 11):
                    for upper_split in range(0, 11):
                        if lower_split<=upper_split:
                            job.addChildJobFn(
                                setup_lrp,
                                "LayerNumRule",
                                UPPER_RULE = upper_rule,
                                MIDDLE_RULE = middle_rule,
                                LOWER_RULE = lower_rule,
                                LAYER_SPLIT = (lower_split/10, upper_split/10)
                            )

if __name__ == "__main__":
    from toil.common import Toil
    from toil.job import Job

    parser = Job.Runner.getDefaultArgumentParser()

    options = parser.parse_args()
    options.logLevel = "DEBUG"
    #options.clean = "always"
    options.targetTime = 1

    with Toil(options) as workflow:
        if not workflow.options.restart:
            job = Job.wrapJobFn(start_toil)
            workflow.start(job)
        else:
            workflow.restart()
