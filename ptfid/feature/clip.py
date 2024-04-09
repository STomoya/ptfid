"""Open-CLIP."""

import open_clip


def get_clip_arch_model(arch: str, weights: str):
    """Create CLIP model."""
    model = open_clip.create_model(arch, weights).visual
    size = model.image_size
    model.output_tokens = False

    return model, size


def get_clip_b32_model(weights: str = 'laion2b_s34b_b79k'):
    """ViT-B-32."""
    return get_clip_arch_model('ViT-B-32', weights)


def get_clip_b16_model(weights: str = 'laion2b_s34b_b88k'):
    """ViT-B-16."""
    return get_clip_arch_model('ViT-B-16', weights)


def get_clip_l14_model(weights: str = 'datacomp_xl_s13b_b90k'):
    """ViT-L-14."""
    return get_clip_arch_model('ViT-L-14', weights)


def get_clip_bigG14_model(weights: str = 'laion2b_s39b_b160k'):
    """ViT-bigG-14."""
    return get_clip_arch_model('ViT-bigG-14', weights)


def get_clip_model():
    """Create default model."""
    return get_clip_l14_model()
