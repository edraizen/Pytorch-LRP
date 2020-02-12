import torch
import torch.nn.functional as F

RULES = {}

class Rule(object):
    @classmethod
    def __init_subclass__(cls, layer_class=None, **kwds):
        global RULES
        RULES[cls.__name__.rsplit(".", 1)[-1]] = cls

    @staticmethod
    def relprop(lrp_layer, m, relevance_in, method="e-rule"):
        relevance_in = relevance_in.view(m.out_shape)
        with torch.no_grad():

            weight = F.relu(m.weight).detach()

            if method == "g-rule":
                weight = (lrp_layer.GAMMA*weight).detach()

            Z = lrp_layer.forward_pass(m, m.in_tensor, weight).detach()

            #Resize in and out to be same since deconvolution has different size
            Z, R = lrp_layer.reshape(Z, relevance_in)
            del relevance_in

            eps = torch.FloatTensor([lrp_layer.EPSILON if method == "e-rule" else lrp_layer.EPS])

            Z += eps.to(Z.device)
            del eps

            S = R / Z
            del Z, R

            C = lrp_layer.backward_pass(m, S, weight)
            del S, weight

            X, C = lrp_layer.reshape(m.in_tensor, C)
            R = X * C
            del X, C

        return R.detach()

class ERule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        return Rule.relprop(lrp_layer, m, relevance_in, method="e-rule")

class ZeroRule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        return Rule.relprop(lrp_layer, m, relevance_in, method="0-rule")

class GRule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        return Rule.relprop(lrp_layer, m, relevance_in, method="g-rule")

class LayerNumRule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in, nlayers):
        layer_level = 1-(m.LAYER_NUM/nlayers)
        if layer_level < 0.2:
            #Upper layers
            print("Running Rule 0")
            return ZeroRule.relprop(lrp_layer, m, relevance_in)
        elif 0.2<=layer_level<0.5:
            #Middle layers
            print("Running Rule E")
            return ERule.relprop(lrp_layer, m, relevance_in)
        else:
            #Lower layers
            print("Running Rule G")
            return GRule.relprop(lrp_layer, m, relevance_in)

class ZBetaRule(Rule):
    """Z-beta rule from https://github.com/albermax/innvestigate/blob/master/innvestigate/analyzer/relevance_based/relevance_rule.py"""
    MIN = [0]
    MAX = [1]

    @staticmethod
    def relprop(lrp_layer, m, relevance_in):

        with torch.no_grad():
            if len(ZBetaRule.MIN) == 1:
                min_ = ZBetaRule.MIN*m.in_shape[1]
            elif len(ZBetaRule.MIN) == m.in_shape:
                min_ = ZBetaRule.MIN
            else:
                raise ValueError("InputLayer MIN must be of length 1 of size of features")

            if len(ZBetaRule.MAX) == 1:
                max_ = ZBetaRule.MIN*m.in_shape[1]
            elif len(ZBetaRule.MIN) == m.in_shape:
                min_ = ZBetaRule.MAX
            else:
                raise ValueError("InputLayer MAX must be of length 1 of size of features")

            low = torch.FloatTensor(min_).repeat(m.in_shape[0], 1).to(m.in_tensor[1].device)
            high = torch.FloatTensor(max_).repeat(m.in_shape[0], 1).to(m.in_tensor[1].device)

            regular_forward = lrp_layer.forward_pass(m, m.in_tensor)
            low_forward = lrp_layer.forward_pass(m, [m.in_tensor[0], low])
            high_forward = lrp_layer.forward_pass(m, [m.in_tensor[0], high])
            Z = regular_forward-(low_forward+high_forward)

            S = relevance_in/(Z+lrp_layer.EPS)

            tmpA = m.in_tensor[1]*lrp_layer.backward_pass(m, m.in_tensor[1]+regular_forward+S)

            tmpB = low*lrp_layer.backward_pass(m, m.in_tensor[1]+low_forward+S)

            tmpC = high*lrp_layer.backward_pass(m, m.in_tensor[1]+high_forward+S)

            R = tmpA-(tmpB+tmpC)

            del max_, min_, low, high, regular_forward, low_forward, high_forward, Z, S,
            del tmpA, tmpB, tmpC

            return R.detach()

class BRule(Rule):
    @staticmethod
    def relprop(self, m, relevance_in):
        if float(beta) in (-1., 0):
            which = "positive" if beta == -1 else "negative"
            which_opp = "negative" if beta == -1 else "positive"
            print("WARNING: With the chosen beta value, "
                  "only " + which + " contributions "
                  "will be taken into account.\nHence, "
                  "if in any layer only " + which_opp +
                  " contributions exist, the "
                  "overall relevance will not be conserved.\n")

        with torch.no_grad():
            out_c, in_c = m.weight.size()
            w = m.weight.repeat((4, 1))
            # First and third channel repetition only contain the positive weights
            w[:out_c][w[:out_c] < 0] = 0
            w[2 * out_c:3 * out_c][w[2 * out_c:3 * out_c] < 0] = 0
            # Second and fourth channel repetition with only the negative weights
            w[1 * out_c:2 * out_c][w[1 * out_c:2 * out_c] > 0] = 0
            w[-out_c:][w[-out_c:] > 0] = 0

            # Repeat across channel dimension (pytorch always has channels first)
            m.in_tensor = m.in_tensor.repeat((1, 4))
            m.in_tensor[:, :in_c][m.in_tensor[:, :in_c] < 0] = 0
            m.in_tensor[:, -in_c:][m.in_tensor[:, -in_c:] < 0] = 0
            m.in_tensor[:, 1 * in_c:3 * in_c][m.in_tensor[:, 1 * in_c:3 * in_c] > 0] = 0

            # Normalize such that the sum of the individual importance values
            # of the input neurons divided by the norm
            # yields 1 for an output neuron j if divided by norm (v_ij in paper).
            # Norm layer just sums the importance values of the inputs
            # contributing to output j for each j. This will then serve as the normalization
            # such that the contributions of the neurons sum to 1 in order to
            # properly split up the relevance of j amongst its roots.

            norm_shape = m.out_shape
            norm_shape[1] *= 4
            norm = torch.zeros(norm_shape).to(self.relevance_in.device)

            for i in range(4):
                norm[:, out_c * i:(i + 1) * out_c] = lrp_layer.forward_pass(
                    m.in_tensor[:, in_c * i:(i + 1) * in_c], w[out_c * i:(i + 1) * out_c])

            # Double number of output channels for positive and negative norm per
            # channel.
            norm_shape[1] = norm_shape[1] // 2
            new_norm = torch.zeros(norm_shape).to(self.innvestigator.device)
            new_norm[:, :out_c] = norm[:, :out_c] + norm[:, out_c:2 * out_c]
            new_norm[:, out_c:] = norm[:, 2 * out_c:3 * out_c] + norm[:, 3 * out_c:]
            norm = new_norm

            # Some 'rare' neurons only receive either
            # only positive or only negative inputs.
            # Conservation of relevance does not hold, if we also
            # rescale those neurons by (1+beta) or -beta.
            # Therefore, catch those first and scale norm by
            # the according value, such that it cancels in the fraction.

            # First, however, avoid NaNs.
            mask = norm == 0
            # Set the norm to anything non-zero, e.g. 1.
            # The actual inputs are zero at this point anyways, that
            # is why norm is zero in the first place.
            norm[mask] = 1
            # The norm in the b-rule has shape (N, 2*out_c, *spatial_dims).
            # The first out_c block corresponds to the positive norms,
            # the second out_c block corresponds to the negative norms.
            # We find the rare neurons by choosing those nodes per channel
            # in which either the positive norm ([:, :out_c]) is zero, or
            # the negative norm ([:, :out_c]) is zero.
            rare_neurons = (mask[:, :out_c] + mask[:, out_c:])

            # Also, catch new possibilities for norm == zero to avoid NaN..
            # The actual value of norm again does not really matter, since
            # the pre-factor will be zero in this case.

            norm[:, :out_c][rare_neurons] *= 1 if self.innvestigator.beta == -1 else 1 + self.innvestigator.beta
            norm[:, out_c:][rare_neurons] *= 1 if self.innvestigator.beta == 0 else -self.innvestigator.beta
            # Add stabilizer term to norm to avoid numerical instabilities.
            norm += self.eps * torch.sign(norm)
            input_relevance = relevance_in.squeeze(dim=-1).repeat(1, 4)
            input_relevance[:, :2*out_c] *= (1+self.beta)/norm[:, :out_c].repeat(1, 2)
            input_relevance[:, 2*out_c:] *= -self.beta/norm[:, out_c:].repeat(1, 2)
            inv_w = w.t()
            relevance_out = torch.zeros_like(m.in_tensor)
            for i in range(4):
                relevance_out[:, i*in_c:(i+1)*in_c] = lrp_layer.backward_pass(
                    input_relevance[:, i*out_c:(i+1)*out_c],
                    inv_w[:, i*out_c:(i+1)*out_c])

            relevance_out *= m.in_tensor

            relevance_out = sum([relevance_out[:, i*in_c:(i+1)*in_c] for i in range(4)])

            del sum_weights, input_relevance, norm, rare_neurons, \
                mask, new_norm, m.in_tensor, w, inv_w
