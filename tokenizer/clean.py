def clean(code):
    return "\n".join(line.rstrip() for line in code.replace("\r\n", "\n").split("\n"))
