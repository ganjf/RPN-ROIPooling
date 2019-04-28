import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .config import cfg
from .generate_anchors_global import generate_anchors_global
from .anchor_target_layer_cpu import anchor_target_layer
from .proposal_layer import proposal_layer




class RPN(nn.Module):
  def __init__(self, net):
    super(RPN, self).__init__()

    self._network = net

    self.cross_entropy = None
    self.loss_box = None
    self.cache_dict = {}


  def _init_network(self):
    self._network._init_network()
    self.rpn_conv = nn.Conv2d(self._network._channels['head'], 512, (3, 3), padding=1)
    self.rpn_score = nn.Conv2d(512, self._network._num_anchors * 2, (1, 1))
    self.rpn_bbox = nn.Conv2d(512, self._network._num_anchors * 4, (1, 1))

  def forward(self, im_data, im_info, gt_boxes=None):
    feature = self._network._image_to_head(im_data)
    rpn_feature = self.rpn_conv(feature)
    # cls
    # n a*2 h w
    rpn_cls_score = self.rpn_score(rpn_feature)
    # n 2 a*h w
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2)
    # n 2 a*h w
    rpn_cls_prob = F.softmax(rpn_cls_score_reshape, 1)
    # n a*2 h w  to  n h w a*2
    rpn_cls_prob_final = self._reshape_layer(rpn_cls_prob, self._network._num_anchors * 2).permute(0, 2, 3, 1).contiguous()

    # bbox
    rpn_bbox_score = self.rpn_bbox(rpn_feature)
    rpn_bbox_score = rpn_bbox_score.permute(0, 2, 3, 1).contiguous()

    # generate anchors
    self._generate_anchors(rpn_cls_score)
    rois, scores = self._region_proposal(rpn_cls_prob_final, rpn_bbox_score, im_info)

    # generating training labels and build the rpn loss
    if self.training:
      assert gt_boxes is not None
      rpn_data = self._anchor_target_layer(rpn_cls_score, gt_boxes, im_info)
      # rpn_data --> rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
      self.cross_entropy, self.loss_box = self._build_loss(rpn_cls_score_reshape, rpn_bbox_score, rpn_data)

    return rois, scores, feature


  @property
  def loss(self):
    return self.cross_entropy + self.loss_box * cfg.TRAIN.LOSS_RATIO

  def _build_loss(self, rpn_cls_score_reshape, rpn_bbox_score, rpn_data, sigma_rpn=3):
    rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
    rpn_label = rpn_data[0].view(-1)

    # cls loss
    rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze())
    assert rpn_keep.numel() == cfg.TRAIN.RPN_BATCHSIZE
    if cfg.CUDA_IF:
      rpn_keep = rpn_keep.cuda()
    rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
    rpn_label = torch.index_select(rpn_label, 0, rpn_keep)
    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

    rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
    rpn_loss_box = self._smooth_l1_loss(rpn_bbox_score, rpn_bbox_targets, rpn_bbox_inside_weights,
                                        rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
    return rpn_cross_entropy, rpn_loss_box

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

  def _reshape_layer(self, x, d):
    '''
    :param x: n [a1 b1,a2 b2] h w
    :param d: d
    :return: n 2 a*h w
    '''
    input_shape = x.size()
    x = x.view(
      input_shape[0],
      int(d),
      int(float(input_shape[1] * input_shape[2]) / float(d)),
      input_shape[3]
    )
    return x

  def _generate_anchors(self, rpn_cls_score):
    # anchors [A*K, 4]
    #feat_stride is (size of origin image)/(size of feature map)
    anchors = generate_anchors_global(
      feat_stride=self._network._feat_stride[0],
      height=rpn_cls_score.size()[-2],
      width=rpn_cls_score.size()[-1],
      anchor_scales=self._network._anchor_scales,
      anchor_ratios=self._network._anchor_ratios
    )
    self._anchors = Variable(torch.from_numpy(anchors)).float()
    if cfg.CUDA_IF:
      self._anchors = self._anchors.cuda()
    self.cache_dict['anchors_cache'] = self._anchors

  def _region_proposal(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
    cfg_key = 'TRAIN' if self.training else 'TEST'
    rois, rpn_scores = proposal_layer(rpn_cls_prob=rpn_cls_prob_reshape, rpn_bbox_pred=rpn_bbox_pred,
                                      im_info=im_info, cfg_key=cfg_key,
                                      _feat_stride=self._network._feat_stride,
                                      anchors=self._anchors,
                                      num_anchors=self._network._num_anchors)
    self.cache_dict['rois'] = rois
    self.cache_dict['rpn_scores'] = rpn_scores
    return rois, rpn_scores

  def _anchor_target_layer(self, rpn_cls_score, gt_boxes, im_info):
    rpn_cls_score = rpn_cls_score.data
    gt_boxes = gt_boxes.data.cpu().numpy()
    all_anchors = self._anchors.data.cpu().numpy()
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      anchor_target_layer(rpn_cls_score=rpn_cls_score, gt_boxes=gt_boxes, im_info=im_info,
                          _feat_stride=self._network._feat_stride,
                          all_anchors=all_anchors,
                          num_anchors=self._network._num_anchors)
    rpn_labels = self.np_to_variable(rpn_labels, is_cuda=cfg.CUDA_IF, dtype=torch.LongTensor)
    rpn_bbox_targets = self.np_to_variable(rpn_bbox_targets, is_cuda=cfg.CUDA_IF)
    rpn_bbox_inside_weights = self.np_to_variable(rpn_bbox_inside_weights, is_cuda=cfg.CUDA_IF)
    rpn_bbox_outside_weights = self.np_to_variable(rpn_bbox_outside_weights, is_cuda=cfg.CUDA_IF)
    self.cache_dict['rpn_labels'] = rpn_labels
    self.cache_dict['rpn_bbox_targets'] = rpn_bbox_targets
    self.cache_dict['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
    self.cache_dict['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


  def np_to_variable(self, x, is_cuda=True, dtype=torch.FloatTensor, requires_grad=False):
    v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
    if is_cuda:
      v = v.cuda()
    return v