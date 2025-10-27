try:
    import nebula_patch
except Exception as e:
    import traceback
    print("Error importing nebula_patch: ", e, traceback.format_exc())
