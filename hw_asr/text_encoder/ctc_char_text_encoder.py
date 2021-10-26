from typing import List, Tuple

from ctcdecode import CTCBeamDecoder
import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from hw_asr.utils.language_model import prepare_lm


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.lm_path = prepare_lm()
        self.alphabet = [self.EMPTY_TOK] + alphabet
        self.ctc_beam_decoder = CTCBeamDecoder(
            self.alphabet,
            model_path=self.lm_path,
            alpha=0.5,
            beta=1.0,
            beam_width=100,
            log_probs_input=True
        )
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        res = ""
        for i, index in enumerate(inds):
            ind = int(index)
            if i > 0 and ind == inds[i - 1] or ind == 0:
                continue
            else:
                res += self.ind2char[ind]
        return res

    def ctc_beam_search(self, log_probs: torch.tensor, only_top_beam: bool = True) -> str:
        beam_results, beam_scores, timesteps, out_lens = self.ctc_beam_decoder.decode(log_probs)
        if only_top_beam:
            return ''.join([self.ind2char[int(i)] for i in beam_results[0][0][:out_lens[0][0]]])
        results = []
        for index in range(beam_results.size(1)):
            results.append(''.join([self.ind2char[int(i)] for i in beam_results[0][index][:out_lens[0][index]]]))
        return results
