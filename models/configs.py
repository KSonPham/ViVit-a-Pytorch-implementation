import ml_collections


def get_vb16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.spatial = ml_collections.ConfigDict()
    config.spatial.transformer = ml_collections.ConfigDict()
    config.spatial.transformer.mlp_dim = 3072
    config.spatial.transformer.num_heads = 12
    config.spatial.transformer.num_layers = 12
    config.spatial.transformer.attention_dropout_rate = 0.0
    config.spatial.transformer.dropout_rate = 0.1
    config.spatial.hidden_size = 768
    
    config.temporal = ml_collections.ConfigDict()
    config.temporal.transformer = ml_collections.ConfigDict()
    config.temporal.transformer.mlp_dim = 3072
    config.temporal.transformer.num_heads = 12
    config.temporal.transformer.num_layers = 4
    config.temporal.transformer.attention_dropout_rate = 0.0
    config.temporal.transformer.dropout_rate = 0.1
    config.temporal.hidden_size = 768
    
    
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config

