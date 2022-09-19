from collections import OrderedDict

DEFAULT_FUSION_PATTERNS = OrderedDict()


def register_fusion_pattern(pattern):
    def insert(fn):
        if pattern in DEFAULT_FUSION_PATTERNS.keys():
            raise
        DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn

    return insert


def get_default_fusion_patterns():
    return DEFAULT_FUSION_PATTERNS
