# GeneralizedRCNN(
#   (backbone): FPN(
#     (fpn_lateral2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#     (fpn_output2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (fpn_lateral3): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
#     (fpn_output3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (fpn_lateral4): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
#     (fpn_output4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (fpn_lateral5): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
#     (fpn_output5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (top_block): LastLevelMaxPool()
#     (bottom_up): ResNet(
#       (stem): BasicStem(
#         (conv1): Conv2d(
#           3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
#           (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#         )
#       )
#       (res2): Sequential(
#         (0): BottleneckBlock(
#           (shortcut): Conv2d(
#             64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv1): Conv2d(
#             64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#         )
#         (1): BottleneckBlock(
#           (conv1): Conv2d(
#             256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#         )
#         (2): BottleneckBlock(
#           (conv1): Conv2d(
#             256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#         )
#       )
#       (res3): Sequential(
#         (0): BottleneckBlock(
#           (shortcut): Conv2d(
#             256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#           (conv1): Conv2d(
#             256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
#             (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#         )
#         (1): BottleneckBlock(
#           (conv1): Conv2d(
#             512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#         )
#         (2): BottleneckBlock(
#           (conv1): Conv2d(
#             512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#         )
#         (3): BottleneckBlock(
#           (conv1): Conv2d(
#             512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#         )
#       )
#       (res4): Sequential(
#         (0): BottleneckBlock(
#           (shortcut): Conv2d(
#             512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False
#             (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
#           )
#           (conv1): Conv2d(
#             512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
#           )
#         )
#         (1): BottleneckBlock(
#           (conv1): Conv2d(
#             1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
#           )
#         )
#         (2): BottleneckBlock(
#           (conv1): Conv2d(
#             1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
#           )
#         )
#         (3): BottleneckBlock(
#           (conv1): Conv2d(
#             1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
#           )
#         )
#         (4): BottleneckBlock(
#           (conv1): Conv2d(
#             1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
#           )
#         )
#         (5): BottleneckBlock(
#           (conv1): Conv2d(
#             1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
#           )
#         )
#       )
#       (res5): Sequential(
#         (0): BottleneckBlock(
#           (shortcut): Conv2d(
#             1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False
#             (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
#           )
#           (conv1): Conv2d(
#             1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
#           )
#         )
#         (1): BottleneckBlock(
#           (conv1): Conv2d(
#             2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
#           )
#         )
#         (2): BottleneckBlock(
#           (conv1): Conv2d(
#             2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#           (conv2): Conv2d(
#             512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
#           )
#           (conv3): Conv2d(
#             512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
#             (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
#           )
#         )
#       )
#     )
#   )
#   (proposal_generator): RPN(
#     (anchor_generator): DefaultAnchorGenerator(
#       (cell_anchors): BufferList()
#     )
#     (rpn_head): StandardRPNHead(
#       (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (objectness_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
#       (anchor_deltas): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (roi_heads): StandardROIHeads(
#     (box_pooler): ROIPooler(
#       (level_poolers): ModuleList(
#         (0): ROIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=0, aligned=True)
#         (1): ROIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=0, aligned=True)
#         (2): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
#         (3): ROIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=0, aligned=True)
#       )
#     )
#     (box_head): FastRCNNConvFCHead(
#       (fc1): Linear(in_features=12544, out_features=1024, bias=True)
#       (fc2): Linear(in_features=1024, out_features=1024, bias=True)
#     )
#     (box_predictor): FastRCNNOutputLayers(
#       (cls_score): Linear(in_features=1024, out_features=81, bias=True)
#       (bbox_pred): Linear(in_features=1024, out_features=320, bias=True)
#     )
#   )
# )
