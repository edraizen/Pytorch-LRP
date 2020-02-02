

from .modules import ALLOWED_LAYERS, ALLOWED_LAYERS_BY_NAME, ALLOWED_PASS_LAYERS, \
    AVAILABLE_METHODS, LRPPassLayer, Flatten

class RelevancePropagator(object):
    """
    Class for computing the relevance propagation and supplying
    the necessary forward hooks for all layers.
    """

    def __init__(self, lrp_exponent, beta, method, epsilon, device, ignore_unsupported_layers=False):
        self.device = device
        self.layer = None
        self.p = lrp_exponent
        self.beta = beta
        self.eps = epsilon
        self.warned_log_softmax = False
        self.module_list = []
        if method not in AVAILABLE_METHODS:
            raise NotImplementedError("Only methods available are: " +
                                      str(AVAILABLE_METHODS))
        self.method = method
        self.ignore_unsupported_layers = ignore_unsupported_layers

    def reset_module_list(self):
        """
        The module list is reset for every evaluation, in change the order or number
        of layers changes dynamically.

        Returns:
            None

        """
        self.module_list = []
        # Try to free memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __get_lrp_layer(self, layer):
        if isinstance(layer, ALLOWED_PASS_LAYERS):
            return LRPPassLayer

        try:
            #Class is key
            return ALLOWED_LAYERS_BY_NAME[layer.__class__]
        except KeyError:
            try:
                #Full name of class in allowed layers => manually mapped
                return ALLOWED_LAYERS_BY_NAME[layer.__class__.__name__]
            except KeyError:
                try:
                    #Check if the last parts of name match
                    name = layer.__class__.__name__.rsplit(".", 1)[-1]
                    return ALLOWED_LAYERS_BY_NAME[name]
                except KeyError:
                    if self.ignore_unsupported_layers:
                        return LRPPassLayer
                    else:
                        raise NotImplementedError("The network contains layers that"
                                                  " are currently not supported {0:s}".format(str(layer)))

    def compute_propagated_relevance(self, layer, relevance):
        """
        This method computes the backward pass for the incoming relevance
        for the specified layer.

        Args:
            layer: Layer to be reverted.
            relevance: Incoming relevance from higher up in the network.

        Returns:
            The

        """
        lrp_layer = self.__get_lrp_layer(layer)(self, self.method)
        return lrp_layer.relprop(layer, relevance)

    def get_layer_fwd_hook(self, layer):
        """
        Each layer might need to save very specific data during the forward
        pass in order to allow for relevance propagation in the backward
        pass. For example, for max_pooling, we need to store the
        indices of the max values. In convolutional layers, we need to calculate
        the normalizations, to ensure the overall amount of relevance is conserved.

        Args:
            layer: Layer instance for which forward hook is needed.

        Returns:
            Layer-specific forward hook.

        """
        lrp_layer = self.__get_lrp_layer(layer)(self, self.method)
        return lrp_layer.forward
