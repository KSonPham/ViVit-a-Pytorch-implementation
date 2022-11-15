import ml_collections


def get_vb16_config():
    """Returns the ViT-B/16 configuration."""
    attention_dropout = 0.1
    drop_out = 0.1
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.classifier = 'token'
    config.representation_size = None


    config.spatial = ml_collections.ConfigDict()
    config.spatial.transformer = ml_collections.ConfigDict()
    config.spatial.transformer.mlp_dim = 3072
    config.spatial.transformer.num_heads = 12
    config.spatial.transformer.num_layers = 12
    config.spatial.transformer.attention_dropout_rate = attention_dropout
    config.spatial.transformer.dropout_rate = drop_out
    config.spatial.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.spatial.hidden_size = config.hidden_size

    config.temporal = ml_collections.ConfigDict()
    config.temporal.transformer = ml_collections.ConfigDict()
    config.temporal.transformer.mlp_dim = 3072
    config.temporal.transformer.num_heads = 12
    config.temporal.transformer.num_layers = 4
    config.temporal.transformer.attention_dropout_rate = attention_dropout
    config.temporal.transformer.dropout_rate = drop_out
    config.temporal.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.temporal.hidden_size = config.hidden_size



    return config


def get_vb16_config_small():
    """Returns the ViT-B/16 Small 6 layers configuration."""
    attention_dropout = 0.1
    drop_out = 0.1
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.classifier = 'token'
    config.representation_size = None


    config.spatial = ml_collections.ConfigDict()
    config.spatial.transformer = ml_collections.ConfigDict()
    config.spatial.transformer.mlp_dim = 3072
    config.spatial.transformer.num_heads = 12
    config.spatial.transformer.num_layers = 6
    config.spatial.transformer.attention_dropout_rate = attention_dropout
    config.spatial.transformer.dropout_rate = drop_out
    config.spatial.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.spatial.hidden_size = config.hidden_size

    config.temporal = ml_collections.ConfigDict()
    config.temporal.transformer = ml_collections.ConfigDict()
    config.temporal.transformer.mlp_dim = 3072
    config.temporal.transformer.num_heads = 12
    config.temporal.transformer.num_layers = 4
    config.temporal.transformer.attention_dropout_rate = attention_dropout
    config.temporal.transformer.dropout_rate = drop_out
    config.temporal.patches = ml_collections.ConfigDict({'size': (16, 16, 2)})
    config.temporal.hidden_size = config.hidden_size

    return config
