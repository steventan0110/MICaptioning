# Beam search for a single batch, inspired by code public on github: https://github.com/kh-kim/simple-nmt/blob/master/simple_nmt/search.py
from operator import itemgetter
import torch
import torch.nn as nn

LENGTH_PENALTY = .2
MIN_LENGTH = 5


class SingleBeamSearchBoard():
    def __init__(
        self,
        device,
        tokenizer,
        prev_status_config,
        beam_size=5,
        max_length=255,
    ):
        self.beam_size = beam_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        # To put data to same device.
        self.device = device
        # Inferred word index for each time-step. For now, initialized with initial time-step.
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + self.tokenizer.bos]
        # Beam index for selected word index, at each time-step.
        self.beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        # Cumulative log-probability for each beam. Set 1 0 because we need the model to explore top-k in the first run
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')]*(beam_size-1)).to(self.device)]
        # 1 if it is done else 0
        self.masks = [torch.BoolTensor(beam_size).zero_().to(self.device)]

        self.prev_status = {}
        self.batch_dims = {}
        for prev_status_name, data in prev_status_config.items():
            self.prev_status[prev_status_name] = torch.cat([data]*beam_size, dim=1)

        self.current_time_step = 0
        self.done_cnt = 0

    def get_length_penalty(
        self,
        length,
        alpha=LENGTH_PENALTY,
        min_length=MIN_LENGTH,
    ):
        p = ((min_length + 1) / (min_length + length))**alpha
        return p

    def is_done(self):
        # Return 1, if we had EOS more than 'beam_size'-times.
        if self.done_cnt >= self.beam_size:
            return 1
        return 0

    def get_batch(self):
        """
        y_hat: beam_size x 1
        prev_status: (beam_size x length x hidden_size) for each layer in RNN
        """
        y_hat = self.word_indice[-1].unsqueeze(-1)
        return y_hat, self.prev_status

    def collect_result(self, y_hat, prev_status):
        """
        y_hat: beam_size x 1 x output_size (vocab size)
        prev_hidden and prev_cell: 1 x beam_size x hidden_size
        """
        self.current_time_step += 1
        output_size = y_hat.size(-1)

        # already finished sentence has cumulative prob = -inf
        cumulative_prob = self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf'))
        cumulative_prob = y_hat + cumulative_prob.view(-1, 1, 1).expand(self.beam_size, 1, output_size)
        top_log_prob, top_indice = cumulative_prob.view(-1).sort(descending=True)
        top_log_prob, top_indice = top_log_prob[:self.beam_size], top_indice[:self.beam_size]

        self.word_indice += [top_indice.fmod(output_size)]
        self.beam_indice += [top_indice.div(float(output_size)).long()]

        # debug usage:
        # print(self.word_indice)
        # print(self.beam_indice)
        # print()
        # Add results to history boards.
        self.cumulative_probs += [top_log_prob]
        self.masks += [torch.eq(self.word_indice[-1], self.tokenizer.eos)] # Set finish mask if we got EOS.
        # Calculate a number of finished beams.
        self.done_cnt += self.masks[-1].float().sum()
        
        for prev_status_name, data in prev_status.items():
            self.prev_status[prev_status_name] = torch.index_select(
                data,
                dim=1, # index select along dim=1
                index=self.beam_indice[-1]
            ).contiguous()
     

    def get_n_best(self, n=1, length_penalty=0):
        sentences, probs, founds = [], [], []

        for t in range(len(self.word_indice)):  # for each time-step,
            for b in range(self.beam_size):  # for each beam,
                if self.masks[t][b] == 1:  # if we had EOS on this time-step and beam,
                    # Take a record of penaltified log-proability.
                    probs += [self.cumulative_probs[t][b] * self.get_length_penalty(t, alpha=length_penalty)]
                    founds += [(t, b)]

        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'): # If this beam does not have EOS,
                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b] * self.get_length_penalty(len(self.cumulative_probs),
                                                                                     alpha=length_penalty)]
                    founds += [(t, b)]
        #print(founds, probs)
      
        # Sort and take n-best.
        sorted_founds_with_probs = sorted(
            zip(founds, probs),
            key=itemgetter(1),
            reverse=True,
        )[:n]
        probs = []

        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []
            # Trace from the end using backpointer, ignore bos by setting end index=0
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b].item()] + sentence
                b = self.beam_indice[t][b]

            sentences += sentence
            probs.append(prob.item())

        return sentences, probs