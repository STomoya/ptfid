"""timm."""

import timm


def get_timm_model(model_name: str):
    """Fet timm model and setup for feature extraction."""
    try:
        model = timm.create_model(model_name=model_name, pretrained=True)
    except RuntimeError:
        import pprint

        print('Models with similar names:')
        pprint.pprint(timm.list_models(f'*{model_name}*', pretrained=True))
        raise
    model.reset_classifier(num_classes=0)

    size = model.pretrained_cfg.get('input_size')
    if size is None:
        raise Exception(f'pretrained_cfg does not contain "input_size" key. {model.pretrained_cfg}')
    size = size[1:]

    return model, size
