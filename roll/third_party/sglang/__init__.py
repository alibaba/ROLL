import sglang as sgl

patch = None
if sgl.__version__ == '0.4.6.post4':
    from roll.third_party.sglang import v046post4_patch
    patch = v046post4_patch
elif sgl.__version__ == '0.4.6.post1':
    from roll.third_party.sglang import v046post1_patch
    patch = v046post1_patch
elif sgl.__version__ == '0.4.10.post2':
    from roll.third_party.sglang import v0410post2_patch
    patch = v0410post2_patch
else:
     raise NotImplementedError(f"Scale aligner version sglang:{sgl.__version__} is not supported.")