def fill_in_the_middle(text, rng):
    lines = text.split("\n")
    if len(lines) < 4:
        return text

    start, end = sorted(rng.sample(range(len(lines)+1), 2))
    pre = "".join(l + "\n" for l in lines[:start])
    mid = "".join(l + "\n" for l in lines[start:end])
    suf = "\n".join(lines[end:])

    if end == len(lines) and mid.endswith("\n"):
        mid = mid[:-1]

    return "<|fim_prefix|>" + pre + "<|fim_suffix|>" + suf + "<|fim_middle|>" + mid


def unfill_in_the_middle(text):
    if not text.startswith("<|fim_prefix|>"):
        return text

    pre, rest = text[len("<|fim_prefix|>"):].split("<|fim_suffix|>", 1)
    suf, mid  = rest.split("<|fim_middle|>", 1)
    return pre + mid + suf