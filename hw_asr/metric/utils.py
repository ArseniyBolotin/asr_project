# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if target_text == '':
        if predicted_text == '':
            return 0.
        return float('Inf')
    dist = editdistance.distance(target_text.split(), predicted_text.split())
    return dist / len(target_text.split())


def calc_wer(target_text, predicted_text) -> float:
    if target_text == '':
        if predicted_text == '':
            return 0.
        return float('Inf')
    dist = editdistance.distance(target_text, predicted_text)
    return dist / len(target_text)
