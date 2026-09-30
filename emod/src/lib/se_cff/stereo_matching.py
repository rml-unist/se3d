import torch
import torch.nn as nn
import torch.nn.functional as F

from .refinement import StereoDRNetRefinement

from .feature_extractor import FeatureExtractor
from .cost import CostVolumePyramid,CostVolume
from .aggregation import AdaptiveAggregation
from .estimation import DisparityEstimationPyramid

class StereoMatchingNetwork(nn.Module):
    def __init__(self, max_disp,
                 in_channels=3,
                 num_downsample=2,
                 no_mdconv=False,
                 feature_similarity='correlation',
                 #feature_similarity= 'concat',
                 num_scales=3,
                 num_fusions=6,
                 deformable_groups=2,
                 mdconv_dilation=2,
                 no_intermediate_supervision=False,
                 num_stage_blocks=1,
                 num_deform_blocks=3,
                 refine_channels=None):
        super(StereoMatchingNetwork, self).__init__()

        refine_channels = in_channels if refine_channels is None else refine_channels
        self.num_downsample = num_downsample
        self.num_scales = num_scales

        # Feature extractor
        self.feature_extractor = FeatureExtractor(in_channels=in_channels)
        max_disp = max_disp // 3

        # Cost volume construction
        self.cost_volume_constructor = CostVolumePyramid(max_disp, feature_similarity='correlation')
        self.cost_volume_constructor_concat = CostVolumePyramid(max_disp, feature_similarity='concat')
        # Cost aggregation
        self.aggregation = AdaptiveAggregation(max_disp=max_disp,
                                               num_scales=num_scales,
                                               num_fusions=num_fusions,
                                               num_stage_blocks=num_stage_blocks,
                                               num_deform_blocks=num_deform_blocks,
                                               no_mdconv=no_mdconv,
                                               mdconv_dilation=mdconv_dilation,
                                               deformable_groups=deformable_groups,
                                               intermediate_supervision=not no_intermediate_supervision)

        # Disparity estimation
        self.disparity_estimation = DisparityEstimationPyramid(max_disp)

        # Refinement
        refine_module_list = nn.ModuleList()
        for i in range(num_downsample):
            refine_module_list.append(StereoDRNetRefinement(img_channels=refine_channels))

        self.refinement = refine_module_list

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1. / pow(2, self.num_downsample - i - 1)

            if scale_factor == 1.0:
                curr_left_img = left_img
                curr_right_img = right_img
            else:
                curr_left_img = F.interpolate(left_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
                curr_right_img = F.interpolate(right_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
            inputs = (disparity, curr_left_img, curr_right_img)
            disparity = self.refinement[i](*inputs)
            disparity_pyramid.append(disparity)  # [H/2, H]

        return disparity_pyramid

    def forward(self, left_img, right_img):
        left_feature = self.feature_extractor(left_img)
        # print("Feature shape: ", len(left_feature))
        # print("Feature shape: ", left_feature[0].shape)
        # print("Feature shape: ", left_feature[1].shape)
        # print("Feature shape: ", left_feature[2].shape)

        right_feature = self.feature_extractor(right_img)
        
        # print("stack shape: ", right_img.dtype)
        # print("img shape: ", rgb_img.dtype)
        # rgb_feature = self.feature_extractor(rgb_img)
        cost_volume = self.cost_volume_constructor(left_feature, right_feature)
        #cost_volume_1 = self.cost_volume_constructor_concat(left_feature, right_feature)
        #print(cost_volume_1[0].shape) 
        # print("Cost_volume shape: ", len(cost_volume))
        # print("Cost_volume shape: ", cost_volume[0].shape)
        # print("Cost_volume shape: ", cost_volume[1].shape)
        # print("Cost_volume shape: ", cost_volume[2].shape)
        # print("####################type",type(cost_volume))
        


        aggregation = self.aggregation(cost_volume)
        # print("Aggregation shape: ", len(aggregation))
        # print("Aggregation shape: ", aggregation[0].shape)
        # print("Aggregation shape: ", aggregation[1].shape)
        # print("Aggregation shape: ", aggregation[2].shape)

        disparity_pyramid = self.disparity_estimation(aggregation)
        # print("Disparity_pyramid shape: ", disparity_pyramid[0].shape)

        disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])

        return left_feature, aggregation, disparity_pyramid

