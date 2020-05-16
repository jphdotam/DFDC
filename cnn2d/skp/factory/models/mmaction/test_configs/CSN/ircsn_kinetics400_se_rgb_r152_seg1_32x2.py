# model settings
model = dict(
    type='TSN3D',
    backbone=dict(
        type='ResNet_R3D',
        pretrained=None,
        depth=152,
        use_pool1=True,
        block_type='3d-sep'),
    spatial_temporal_module=dict(
        type='SimpleSpatialTemporalModule',
        spatial_type='avg',
        temporal_size=-1,
        spatial_size=-1),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=2048,
        num_classes=400))

train_cfg = None
test_cfg = None
# dataset settings
