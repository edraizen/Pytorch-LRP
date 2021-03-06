import torch
import torch.nn.functional as F

RULES = {}

class Rule(object):
    @classmethod
    def repr(cls):
        return "<{}>".format(cls.__name__)

    @classmethod
    def __init_subclass__(cls, layer_class=None, **kwds):
        global RULES
        RULES[cls.__name__.rsplit(".", 1)[-1]] = cls

    @staticmethod
    def relprop(lrp_layer, m, relevance_in, weight_transform=None, eps=1e-6):
        #relevance_in = relevance_in.view(m.out_shape)
        with torch.no_grad():

            weight = m.weight #F.relu(m.weight) #.detach()

            if weight_transform is not None and callable(weight_transform):
                weight = weight_transform(weight).detach()

            Z1 = lrp_layer.forward_pass(m, m.in_tensor, weight).detach()

            Z, R = lrp_layer.reshape(Z1, relevance_in)
            del relevance_in
            _eps = eps
            eps = torch.sign(Z)*torch.FloatTensor([eps]).to(Z.device)
            eps[eps==0.] += _eps #torch.FloatTensor([eps])

            Z += eps.to(Z.device)
            S = R / Z

            del Z1, Z, R, eps

            C = lrp_layer.backward_pass(m, S, weight)
            del S, weight

            X, C = lrp_layer.reshape(m.in_tensor, C)
            R = X * C
            del X, C

        return R.detach()

class ERule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        return Rule.relprop(lrp_layer, m, relevance_in, eps=lrp_layer.EPSILON)

class ERuleReLU(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        return Rule.relprop(lrp_layer, m, relevance_in, eps=lrp_layer.EPSILON,
            weight_transform=lambda w: w.clamp(min=0))

class ZeroRule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        return Rule.relprop(lrp_layer, m, relevance_in, eps=lrp_layer.EPS)

class ZeroRuleReLU(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        return Rule.relprop(lrp_layer, m, relevance_in, eps=lrp_layer.EPS,
            weight_transform=lambda w: w.clamp(min=0))

class GRule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        weight_transform = lambda w: w*lrp_layer.GAMMA
        return Rule.relprop(lrp_layer, m, relevance_in, weight_transform=weight_transform,
            eps=lrp_layer.EPS)

class GRuleReLU(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        weight_transform = lambda w: (w*lrp_layer.GAMMA).clamp(min=0)
        return Rule.relprop(lrp_layer, m, relevance_in, weight_transform=weight_transform,
            eps=lrp_layer.EPS)

class LayerNumRule(Rule):
    UPPER_RULE = GRule
    MIDDLE_RULE = ERule
    LOWER_RULE = ZeroRule
    LAYER_SPLIT_MIN = 0.2
    LAYER_SPLIT_MAX = 0.6

    @classmethod
    def repr(cls):
        return """<{}:
    UPPER_RULE={},
    MIDDLE_RULE={},
    LOWER_RULE={},
    LAYER_SPLIT_MIN={},
    LAYER_SPLIT_MAX={}
>""".format(cls.__name__, cls.UPPER_RULE, cls.MIDDLE_RULE, cls.LOWER_RULE,
        cls.LAYER_SPLIT_MIN, cls.LAYER_SPLIT_MAX)

    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        layer_level = m.LAYER_NUM/m.N_LAYERS
        # if lrp_layer.FIRST_LAYER:
        #     if m.input_shape[-1] < 4:
        #         return ZBetaRule.relprop(lrp_layer, m, relevance_in)
        #     else:
        #         return WSquareRule.relprop(lrp_layer, m, relevance_in)
        if layer_level < float(LAYER_SPLIT_MIN):
            #Upper layers
            return UPPER_RULE.relprop(lrp_layer, m, relevance_in) #
        elif float(LAYER_SPLIT_MIN)<=float(layer_level<LAYER_SPLIT_MAX):
            #Middle layers
            return MIDDLE_RULE.relprop(lrp_layer, m, relevance_in)
        else:
            #Lower layers
            return ZeroRule.relprop(lrp_layer, m, relevance_in) # Alpha1Beta0

class WSquareRule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in, pow=2):
        weights = lrp_layer.weight.pow(pow).detach()
        Y = lrp_layer.forward_pass(m, lrp_layer.in_tensor, weights)

        ones = torch.ones(lrp_layer.in_shape)
        Z = lrp_layer.forward_pass(m, ones, weights)
        del ones

        S = S/(Z+torch.sign(Z).to(Z.device)*lrp_layer.EPS)
        del Z

        R = lrp_layer.backward_pass(m, lrp_layer.in_tensor+Y+S, weights)
        del weights, Y, S

        return R.detach()

class FlatRule(Rule):
    """The flat rule works with weights equal to one and no biases."""
    def relprop(lrp_layer, m, relevance_in):
        return WSquareRule.relprop(lrp_layer, m, relevance_in, pow=0)

class TwoLayerSumRule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in, x1, weight1, x2, weight2):
        Z1 = lrp_layer.forward_pass(m, x1, weight1)
        Z2 = lrp_layer.forward_pass(m, x2, weight2)

        Z = Z1+Z2
        S = relevance_in/(Z+torch.sign(Z).to(Z.device)*lrp_layer.EPS)

        assert 0, """X1 {} {} \n\n\n
Z1 {} {}
Z2 {} {}
Z {} {}
S {} {}""".format(x1.size(), x1, Z1.size(), Z1, Z2.size(), Z2, Z.size(), Z, S.size(), S)

        C_pos = lrp_layer.backward_pass(m, x1+Z1+S, weight1)
        C_neg = lrp_layer.backward_pass(m, x2+Z2+S, weight2)
        del Z1, Z2, Z, S

        R1 = x1*C1
        R2 = x2*C2
        del C1, C2

        R = R1+R2
        del R1, R2

        return R.detach()

class AlphaBetaRule(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in, a=None, b=None):
        a = a if isinstance(a, (float, int)) else getattr(m, "ALPHA", 1)
        b = b if isinstance(b, (float, int)) else getattr(b, "BETA", 0)

        R_pos = Rule.relprop(lrp_layer, m, relevance_in, eps=lrp_layer.EPS, weight_transform=lambda w: w.clamp(min=0))
        R_neg = Rule.relprop(lrp_layer, m, relevance_in, eps=lrp_layer.EPS, weight_transform=lambda w: w.clamp(max=0))

        return a*R_pos-b*R_neg

        # x_pos = (m.in_tensor * (m.in_tensor>0).float()).detach()
        # x_neg = (m.in_tensor * (m.in_tensor<0).float()).detach()
        #
        # weight_pos = (m.weight * (m.weight>0).float()).detach()
        # weight_neg = (m.weight * (m.weight<0).float()).detach()
        #
        # Z1 = lrp_layer.forward_pass(m, x_pos, weight_pos).detach()
        # Z2 = lrp_layer.forward_pass(m, x_neg, weight_neg).detach()
        # Z = Z1+Z2
        #
        # S1 = relevance_in/(Z1+torch.sign(Z1).to(Z1.device)*lrp_layer.EPS)
        # S2 = relevance_in/(Z2+torch.sign(Z2).to(Z2.device)*lrp_layer.EPS)
        #
        # C_pos = lrp_layer.backward_pass(m, S1, weight_pos)
        # C_neg = lrp_layer.backward_pass(m, S2, weight_neg)
        #
        # R_pos = m.in_tensor*C_pos
        # R_neg = m.in_tensor*C_neg
        #
        # R = a*R_pos-b*R_neg

        # Z, R = lrp_layer.reshape(Z1, relevance_in)
        # del relevance_in
        # _eps = eps
        # eps = torch.sign(Z)*torch.FloatTensor([eps]).to(Z.device)
        # eps[eps==0.] += _eps #torch.FloatTensor([eps])
        #
        # Z += eps.to(Z.device)
        #
        # S = R / Z
        #
        # del Z1, Z, R, eps
        #
        # C = lrp_layer.backward_pass(m, S, weight)
        # del S, weight
        #
        # X, C = lrp_layer.reshape(m.in_tensor, C)
        # R = X * C
        # del X, C
        #
        #
        #
        # activator_relevances = TwoLayerSumRule.relprop(lrp_layer, m,
        #     relevance_in, x_pos, weight_pos, x_neg, weight_neg)
        #
        # if b==0:
        #     R = a*activator_relevances.detach()
        # else:
        #     inhibitor_relevances = TwoLayerSumRule.relprop(lrp_layer, m,
        #         relevance_in, x_pos, weight_neg, x_neg, weight_pos)
        #     R = a*activator_relevances-b*inhibitor_relevances
        #     del inhibitor_relevances
        #
        # del relevance_in, x_pos, weight_pos, x_neg, weight_neg, activator_relevances

        return R.detach()

class Alpha2Beta1(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in, a=None, b=None):
        return AlphaBetaRule.relprop(lrp_layer, m, relevance_in, a=2, b=1)


class Alpha1Beta0(Rule):
    @staticmethod
    def relprop(lrp_layer, m, relevance_in, a=None, b=None):
        return AlphaBetaRule.relprop(lrp_layer, m, relevance_in, a=1, b=0)

class ZBetaRule(Rule):
    """Z-beta rule from https://github.com/albermax/innvestigate/blob/master/innvestigate/analyzer/relevance_based/relevance_rule.py"""
    MIN = [0]
    MAX = [1]

    @staticmethod
    def relprop(lrp_layer, m, relevance_in):
        return relevance_in
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

            # z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
            # z -= utils.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
            # z -= utils.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)
            #
            # regular_forward = lrp_layer.forward_pass(m, m.in_tensor)
            # low_forward = lrp_layer.forward_pass(m, m.in_tensor.clamp(min=0))
            # high_forward = lrp_layer.forward_pass(m, m.in_tensor.clamp(min=0))
            # Z = regular_forward-(low_forward+high_forward)


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
