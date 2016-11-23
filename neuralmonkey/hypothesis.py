class Hypothesis(object):
    """A class that represents a single hypothesis in a beam."""

    # pylint: disable=too-many-arguments
    # Maybe the logits can be refactored out (they serve only to compute loss)
    def __init__(self, tokens, log_prob, state, rnn_output=None, logits=None):
        """Construct a new hypothesis object

        Arguments:
            tokens: The list of already decoded tokens
            log_prob: The log probability of the decoded tokens given the model
            state: The last state of the decoder
            rnn_output: The last rnn_output of the decoder
            logits: The list of logits over the vocabulary (in time)
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state
        self._rnn_output = rnn_output
        self._logits = [] if logits is None else logits

    # pylint: disable=too-many-arguments
    # Maybe the logits can be refactored out (they serve only to compute loss)
    def extend(self, token, log_prob, new_state, new_rnn_output, new_logits):
        """Return an extended version of the hypothesis.

        Arguments:
            token: The token to attach to the hypothesis
            log_prob: The log probability of emiting this token
            new_state: The RNN state of the decoder after decoding the token
            new_rnn_output: The RNN output tensor that emitted the token
            new_logits: The logits made from the RNN output
        """
        return Hypothesis(self.tokens + [token],
                          self.log_prob + log_prob,
                          new_state, new_rnn_output,
                          self._logits + [new_logits])

    @property
    def latest_token(self):
        """Get the last token from the hypothesis."""
        return self.tokens[-1]

    @property
    def rnn_output(self):
        """Get the last RNN output"""
        if self._rnn_output is None:
            raise Exception("Getting rnn_output before specifying it")
        return self._rnn_output

    @property
    def runtime_logits(self):
        """Get the sequence of logits (for computing loss)"""
        if self._logits is None:
            raise Exception("Getting logit sequence from empty hypothesis")
        return self._logits

    def __str__(self):
        return ("Hypothesis(log prob = {:.4f}, tokens = {})".format(
            self.log_prob, self.tokens))

    def __eq__(self, other):
        """Check whether two hypotheses are equal"""
        # compare hypotheses lengths
        if len(self.tokens) != len(other.tokens):
            return False
        # check if tokens are the same
        if not all(token_a == token_b for (token_a, token_b) in zip(self.tokens, other.tokens)):
            return False
        # check if log_prob is the same
        if self.log_prob != other.logprob:
            return False
        return True


def sort_hypotheses(hyps, normalize_by_length=True):
    """Sort hypotheses based on log probs and length.

    Args:
        hyps: A list of hypothesis.
        normalize_by_length: Whether to normalize by length
    Returns:
        hyps: A list of sorted hypothesis in reverse log_prob order.
    """
    if normalize_by_length:
        return sorted(hyps, key=lambda h: h.log_prob/len(h.tokens),
                      reverse=True)
    else:
        return sorted(hyps, key=lambda h: h.log_prob, reverse=True)