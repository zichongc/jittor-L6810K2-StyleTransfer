import jittor as jt


def dot_product(a: jt.Var, b: jt.Var, dim=1):
    return jt.sum(a*b, dim=dim)


def cosine_similarity(a: jt.Var, b: jt.Var, dim=1):
    a_norm = jt.norm(a, p=2, dim=dim)
    b_norm = jt.norm(b, p=2, dim=dim)
    return dot_product(a, b, dim=dim) / (a_norm * b_norm)
