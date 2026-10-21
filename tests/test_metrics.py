from livetr.metrics import cer_counts, edit_distance, normalize_hy, wer_counts


def test_edit_distance():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], list("abc")) == 3


def test_normalize_removes_armenian_punctuation_and_case():
    assert normalize_hy("Բարև, ինչպե՞ս ես։") == normalize_hy("բարեւ ինչպես ես")


def test_perfect_and_wrong_word():
    assert wer_counts("Ես ապրում եմ Երևանում։", "ես ապրում եմ երևանում") == (0, 4)
    assert wer_counts("Ես ապրում եմ Երևանում։", "ես ապրում եմ գյումրիում") == (1, 4)


def test_cer_ignores_spaces():
    errors, total = cer_counts("լավ եմ", "լավեմ")
    assert errors == 0 and total == 5
