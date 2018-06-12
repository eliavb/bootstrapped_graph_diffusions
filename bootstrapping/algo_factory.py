import label_propagation
import feature_diffusion

def algo_factory(G, c, inital_seed_set, args, special_params):
  if (args.model == "norm_lp" or args.model == "lp"):
    return label_propagation.LabelPropagationBootstrapped(G, c, inital_seed_set, args, special_params)
  if ("feature_diffusion" in args.model):
    return feature_diffusion.FeatureDiffusionBootstrapped(G, c, inital_seed_set, args, special_params) 
  raise Exception("Unkonw type")