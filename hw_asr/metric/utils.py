# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        if len(predicted_text) == 0:
            return 0.
        return float('Inf')
    dist = editdistance.distance(target_text, predicted_text)
    return dist / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text.split()) == 0:
        if len(predicted_text.split()) == 0:
            return 0.
        return float('Inf')
    dist = editdistance.distance(target_text.split(), predicted_text.split())
    return dist / len(target_text.split())
