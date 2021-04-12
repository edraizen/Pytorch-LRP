import time
import torch
import numpy as np

from .modules import LRPLayer, LRPPassLayer, LRPFunctionLayer, cpu_usage, gpu_usage

class InnvestigateModel(object):
    """
    ATTENTION:
        Currently, innvestigating a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If., for example,
        only the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(self, the_model, lrp_exponent=1, beta=.5, epsilon=1e-6,
                 prediction=None, ignore_unsupported_layers=False, rule=None):
        """
        Model wrapper for pytorch models to 'innvestigate' them
        with layer-wise relevance propagation (LRP) as introduced by Bach et. al
        (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).
        Given a class level probability produced by the model under consideration,
        the LRP algorithm attributes this probability to the nodes in each layer.
        This allows for visualizing the relevance of input pixels on the resulting
        class probability.

        Args:
            the_model: Pytorch model, e.g. a pytorch.nn.Sequential consisting of
                        different layers. Not all layers are supported yet.
            lrp_exponent: Exponent for rescaling the importance values per node
                            in a layer when using the e-rule method.
            beta: Beta value allows for placing more (large beta) emphasis on
                    nodes that positively contribute to the activation of a given node
                    in the subsequent layer. Low beta value allows for placing more emphasis
                    on inhibitory neurons in a layer. Only relevant for method 'b-rule'.
            epsilon: Stabilizing term to avoid numerical instabilities if the norm (denominator
                    for distributing the relevance) is close to zero.
            method: Different rules for the LRP algorithm, b-rule allows for placing
                    more or less focus on positive / negative contributions, whereas
                    the e-rule treats them equally. For more information,
                    see the paper linked above.
        """
        # super(InnvestigateModel, self).__init__()
        self.model = the_model
        self.prediction = prediction
        self.in_tensor = None
        self.r_values_per_layer = None
        self.only_max_score = None


        self.beta = beta
        self.eps = epsilon
        self.p = lrp_exponent
        self.ignore_unsupported_layers = ignore_unsupported_layers

        LRPLayer.LRP_EXPONENT = self.p
        LRPLayer.EPS = self.eps
        LRPLayer.BETA = self.beta
        LRPLayer.ignore_unsupported_layers = self.ignore_unsupported_layers

        if rule is not None:
            LRPLayer.DEFAULT_RULE = rule

        self.hooks = []

        # Parsing the individual model layers
        print("Start unravel", cpu_usage(), gpu_usage())
        setattr(self.model, "LAYER_NUM", 0)
        nlayers = self.register_hooks(self.model)
        LRPLayer.N_LAYERS = nlayers

    def register_hooks(self, parent_module, start_index=1, it=0):
        """
        Recursively unrolls a model and registers the required
        hooks to save all the necessary values for LRP in the forward pass.

        Args:
            parent_module: Model to unroll and register hooks for.

        Returns:
            None

        """
        # print(str(parent_module).split("\n")[0], "Layer num", start_index)
        # setattr(parent_module, "LAYER_NUM", start_index)
        self.hooks = []
        start = False
        for i, mod in enumerate(parent_module.children()):
            lrp_layer = LRPLayer.get(mod, raise_if_unknown=False)
            if list(mod.children()) and (lrp_layer is None or not issubclass(lrp_layer, LRPFunctionLayer)):
                start_index = self.register_hooks(mod, start_index=start_index+i+1)
            else:
                if not isinstance(lrp_layer, LRPPassLayer):
                    fwd = mod.register_forward_hook(lrp_layer.forward_hook)
                    self.hooks.append(fwd)
                    if hasattr(lrp_layer, "backward"):
                        rev = mod.register_backward_hook(lrp_layer.backward_hook)
                        self.hooks.append(rev)
                    if not start and start_index == 1 and not isinstance(lrp_layer, NoCountLRPLayer):
                        lrp_layer.FIRST_LAYER = True
                        start = True

        if start_index ==1:
            import pdb; pdb.set_trace()
        return start_index+i+2 if i>0 else start_index+1

    def __call__(self, in_tensor):
        """
        The innvestigate wrapper returns the same prediction as the
        original model, but wraps the model call method in the evaluate
        method to save the last prediction.

        Args:
            in_tensor: Model input to pass through the pytorch model.

        Returns:
            Model output.
        """
        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor):
        """
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron per layer.

        Args:
            in_tensor: New input for which to predict an output.

        Returns:
            Model prediction
        """
        self.in_tensor = in_tensor
        self.prediction = self.model(in_tensor)
        return self.prediction

    def get_r_values_per_layer(self, parent_module=None):
        if parent_module is None:
            parent_module = self.model

        r_values = []
        for mod in parent_module.children():
            if list(mod.children()):
                r_values += self.get_r_values_per_layer(mod)
                continue
            if not hasattr(mod, "relevance"):
                return []
            r_values.append(mod.relevance)

        if parent_module is None and len(r_values) == 0:
            print("No relevances have been calculated yet, returning None in"
                   " get_r_values_per_layer.")
            return None

        return r_values

    def innvestigate(self, in_tensor=None, no_recalc=False, rel_for_class=None, autoencoder_in=False,
        autoencoder_out=False, rule=None, clean=True):
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        with torch.no_grad():
            # Check if innvestigation can be performed.
            if not self.hooks:
                nlayers = self.register_hooks(self.model)
                LRPLayer.N_LAYERS = nlayers

            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            print("Start InnvestigateModel", cpu_usage(), gpu_usage())

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                if not no_recalc:
                    self.evaluate(in_tensor)
                else:
                    self.in_tensor = in_tensor

            # If no class index is specified, analyze for class
            # with highest prediction.
            if not (autoencoder_in or autoencoder_out):
                if rel_for_class is None:
                    # Default behaviour is innvestigating the output
                    # on an arg-max-basis, if no class is specified.
                    org_shape = self.prediction.size()
                    # Make sure shape is just a 1D vector per batch example.
                    self.prediction = self.prediction.view(org_shape[0], -1)
                    max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                    only_max_score = torch.zeros_like(self.prediction)
                    only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                    relevance = only_max_score.view(org_shape)

                else:
                    org_shape = self.prediction.size()
                    self.prediction = self.prediction.view(org_shape[0], -1)
                    only_max_score = torch.zeros_like(self.prediction)
                    only_max_score[:, rel_for_class] += self.prediction[:, rel_for_class]
                    relevance = only_max_score.view(org_shape)
            elif no_recalc:
                relevance = self.in_tensor
            else:
                relevance = self.in_tensor[1] if autoencoder_in else self.prediction

            lrp_module = LRPLayer.get(self.model)
            lrp_module.NORMALIZE_BEFORE = True
            lrp_module.NORMALIZE_AFTER = True

            if rule is not None:
                lrp_module.DEFAULT_RULE = rule

            torch.cuda.empty_cache()

            relevance_out = lrp_module.relprop_(self.model, relevance, rule=rule, clean=clean)

            torch.cuda.empty_cache()

            del lrp_module

            if False:
                # List to save relevance distributions per layer
                r_values_per_layer = [relevance]
                for layer in rev_model:
                    # Compute layer specific backwards-propagation of relevance values
                    relevance = self.compute_propagated_relevance(layer, relevance)
                    r_values_per_layer.append(relevance.cpu())

                self.r_values_per_layer = r_values_per_layer

            return self.prediction, relevance_out

    def forward(self, in_tensor):
        return self.model.forward(in_tensor)

    def clean(self):
        for hook in self.hooks:
            hook.remove()
        self.prediction = None

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return self.model.extra_repr()
