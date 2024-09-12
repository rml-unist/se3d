
# DATA_SPLIT = {
#     'train': ['phase1','phase1_1','phase2','phase2_1','phase3','phase3_1','phase4','phase5','phase6_1','phase7'],
#     'validation': ['phase1','phase1_1','phase2','phase2_1','phase3','phase3_1','phase4','phase5','phase6_1','phase7'],
#     'trainval': ['phase1_1','phase2_1','phase3_1','phase4','phase5','phase7'],
#     'test': ['phase6'],
#     'none': [],
# }

DATA_SPLIT = {
    'train': [
        'map10/map10_day_rain_moving',
        'map3/map3_night_heavyrain_moving',
        'map10/map10_day_heavyrain_moving',
        'map4/map4_night_heavyrain_moving',
        'map2/map2_night_rain_moving',
        'map10/map10_night_sunny_moving',
        'map7/map7_day_sunny_moving',
        'map5/map5_night_heavyrain_moving',
        'map7/map7_night_sunny_moving',
        'map2/map2_day_sunny_moving',
        'map3/map3_night_rain_moving',
        'map7/map7_night_rain_moving',
        'map4/map4_night_sunny_moving',
        'map6/map6_day_sunny_moving',
        'map1/map1_day_rain_moving',
        'map2/map2_night_sunny_moving',
        'map7/map7_night_heavyrain_moving',
        'map6/map6_night_sunny_moving',
        'map5/map5_day_rain_moving',
        'map1/map1_day_heavyrain_moving',
        'map6/map6_day_rain_moving',
        'map5/map5_day_sunny_moving',
        'map10/map10_night_rain_moving',
        'map5/map5_night_sunny_moving',
        'map3/map3_day_rain_moving',
        'map2/map2_day_heavyrain_moving',
        'map4/map4_night_rain_moving',
        'map1/map1_day_sunny_moving',
        'map6/map6_night_rain_moving',
        'map4/map4_day_rain_moving',
        'map1/map1_night_heavyrain_moving',
        'map3/map3_night_sunny_moving',
    ],
    'val': [
        'map6/map6_day_heavyrain_moving',
        'map4/map4_day_sunny_moving',
        'map1/map1_night_rain_moving',
        'map10/map10_night_heavyrain_moving',
        'map2/map2_night_heavyrain_moving',
        'map3/map3_day_heavyrain_moving',
        'map5/map5_day_heavyrain_moving',
        'map7/map7_day_rain_moving',
    ],
    'trainval': [
        'map10/map10_day_rain_moving',
        'map3/map3_night_heavyrain_moving',
        'map10/map10_day_heavyrain_moving',
        'map4/map4_night_heavyrain_moving',
        'map2/map2_night_rain_moving',
        'map10/map10_night_sunny_moving',
        'map7/map7_day_sunny_moving',
        'map5/map5_night_heavyrain_moving',
        'map7/map7_night_sunny_moving',
        'map2/map2_day_sunny_moving',
        'map3/map3_night_rain_moving',
        'map7/map7_night_rain_moving',
        'map4/map4_night_sunny_moving',
        'map6/map6_day_sunny_moving',
        'map1/map1_day_rain_moving',
        'map2/map2_night_sunny_moving',
        'map7/map7_night_heavyrain_moving',
        'map6/map6_night_sunny_moving',
        'map5/map5_day_rain_moving',
        'map1/map1_day_heavyrain_moving',
        'map6/map6_day_rain_moving',
        'map5/map5_day_sunny_moving',
        'map10/map10_night_rain_moving',
        'map5/map5_night_sunny_moving',
        'map3/map3_day_rain_moving',
        'map2/map2_day_heavyrain_moving',
        'map4/map4_night_rain_moving',
        'map1/map1_day_sunny_moving',
        'map6/map6_night_rain_moving',
        'map4/map4_day_rain_moving',
        'map1/map1_night_heavyrain_moving',
        'map6/map6_day_heavyrain_moving',
        'map4/map4_day_sunny_moving',
        'map1/map1_night_rain_moving',
        'map10/map10_night_heavyrain_moving',
        'map2/map2_night_heavyrain_moving',
        'map3/map3_day_heavyrain_moving',
        'map5/map5_day_heavyrain_moving',
        'map7/map7_day_rain_moving',
        'map3/map3_night_sunny_moving',
    ],
    'test': [
        'map2/map2_day_rain_moving',
        'map4/map4_day_heavyrain_moving',
        'map1/map1_night_sunny_moving',
        'map6/map6_night_heavyrain_moving',
        'map7/map7_day_heavyrain_moving',
        'map5/map5_night_rain_moving',
        'map10/map10_day_sunny_moving',
        'map3/map3_day_sunny_moving',
    ],
    'none': [
    ],
}

# DATA_SPLIT={'train': ['map10/map10_night_heavyrain_moving',
# 'map6/map6_day_rain_moving',
# 'map2/map2_day_sunny_moving',
# 'map5/map5_day_heavyrain_moving',
# 'map2/map2_night_sunny_moving',
# 'map5/map5_day_sunny_moving',
# 'map7/map7_night_sunny_moving',
# 'map5/map5_night_heavyrain_moving',
# 'map10/map10_night_rain_moving',
# 'map2/map2_night_rain_moving',
# 'map3/map3_night_heavyrain_moving',
# 'map2/map2_night_heavyrain_moving',
# 'map5/map5_night_sunny_moving',
# 'map3/map3_day_heavyrain_moving',
# 'map10/map10_day_heavyrain_moving',
# 'map1/map1_day_rain_moving',
# 'map7/map7_day_rain_moving',
# 'map3/map3_day_rain_moving',
# 'map1/map1_day_heavyrain_moving',
# 'map1/map1_night_rain_moving',
# 'map6/map6_day_sunny_moving',
# 'map7/map7_day_sunny_moving',
# 'map4/map4_night_heavyrain_moving',
# 'map4/map4_day_heavyrain_moving',
# 'map2/map2_day_rain_moving',
# 'map5/map5_night_rain_moving',
# 'map1/map1_night_sunny_moving',
# 'map7/map7_day_heavyrain_moving',
# 'map3/map3_day_sunny_moving',
# 'map10/map10_day_rain_moving',
# 'map6/map6_night_sunny_moving',
# 'map10/map10_day_sunny_moving'],
# 'val': [
#     'map7/map7_night_heavyrain_moving',
# 'map2/map2_day_heavyrain_moving',
# 'map5/map5_day_rain_moving',
# 'map4/map4_day_rain_moving',
# 'map7/map7_night_rain_moving',
# 'map6/map6_night_rain_moving',
# 'map10/map10_night_sunny_moving',
# 'map4/map4_night_sunny_moving',
# 'map3/map3_night_rain_moving'
# ],
# 'trainval': ['map10/map10_night_heavyrain_moving',
# 'map6/map6_day_rain_moving',
# 'map2/map2_day_sunny_moving',
# 'map5/map5_day_heavyrain_moving',
# 'map2/map2_night_sunny_moving',
# 'map5/map5_day_sunny_moving',
# 'map7/map7_night_sunny_moving',
# 'map5/map5_night_heavyrain_moving',
# 'map10/map10_night_rain_moving',
# 'map2/map2_night_rain_moving',
# 'map3/map3_night_heavyrain_moving',
# 'map2/map2_night_heavyrain_moving',
# 'map5/map5_night_sunny_moving',
# 'map3/map3_day_heavyrain_moving',
# 'map10/map10_day_heavyrain_moving',
# 'map1/map1_day_rain_moving',
# 'map7/map7_day_rain_moving',
# 'map3/map3_day_rain_moving',
# 'map1/map1_day_heavyrain_moving',
# 'map1/map1_night_rain_moving',
# 'map6/map6_day_sunny_moving',
# 'map7/map7_day_sunny_moving',
# 'map4/map4_night_heavyrain_moving',
# 'map4/map4_day_heavyrain_moving',
# 'map2/map2_day_rain_moving',
# 'map5/map5_night_rain_moving',
# 'map1/map1_night_sunny_moving',
# 'map7/map7_day_heavyrain_moving',
# 'map3/map3_day_sunny_moving',
# 'map10/map10_day_rain_moving',
# 'map6/map6_night_sunny_moving',
# 'map10/map10_day_sunny_moving',
# 'map7/map7_night_heavyrain_moving',
# 'map2/map2_day_heavyrain_moving',
# 'map5/map5_day_rain_moving',
# 'map4/map4_day_rain_moving',
# 'map7/map7_night_rain_moving',
# 'map6/map6_night_rain_moving',
# 'map10/map10_night_sunny_moving',
# 'map4/map4_night_sunny_moving',
# 'map3/map3_night_rain_moving'],
# 'test': ['map1/map1_day_sunny_moving',
# 'map1/map1_night_heavyrain_moving',
# 'map4/map4_night_rain_moving',
# 'map4/map4_day_sunny_moving',
# 'map6/map6_night_heavyrain_moving',
# 'map6/map6_day_heavyrain_moving'],
# 'none': []}

# DATA_SPLIT = {'train': ['map7/map7_night_rain_stop',
#   'map7/map7_night_heavyrain_stop',
#   'map7/map7_day_heavyrain_moving',
#   'map7/map7_day_rain_moving',
#   'map7/map7_night_sunny_both',
#   'map7/map7_night_rain_moving',
#   'map7/map7_night_rain_both',
#   'map7/map7_night_heavyrain_both',
#   'map7/map7_night_sunny_stop'],
#  'validation': [],
#  'trainval': ['map7/map7_night_rain_stop',
#   'map7/map7_night_heavyrain_stop',
#   'map7/map7_day_heavyrain_moving',
#   'map7/map7_day_rain_moving',
#   'map7/map7_night_sunny_both',
#   'map7/map7_night_rain_moving',
#   'map7/map7_night_rain_both',
#   'map7/map7_night_heavyrain_both',
#   'map7/map7_night_sunny_stop'],
#  'test': ['map7/map7_day_sunny_moving',
#   'map7/map7_night_sunny_moving',
#   'map7/map7_night_heavyrain_moving'],
#  'none': []}

# DATA_SPLIT = {'train': ['map7/map7_day_heavyrain_moving',
#   'map7/map7_day_rain_moving',
#   'map7/map7_night_rain_moving',],
#  'validation': [],
#  'trainval': ['map7/map7_day_heavyrain_moving',
#   'map7/map7_day_rain_moving',
#   'map7/map7_night_rain_moving',
#    'map7/map7_night_sunny_moving'],
#  'test': ['map7/map7_day_sunny_moving',
#   'map7/map7_night_heavyrain_moving'],
#  'none': []}

# DATA_SPLIT = {'train': ['map6/map6_7583967482_stop',
#   'map6/map6_day_heavyrain_moving',
#   'map4/map4_night_heavyrain_moving',
#   'map7/map7_126262_stop',
#   'map5/map5_7583967482_stop',
#   'map1/map1_126262_both',
#   'map3/map3_7583967482_both',
#   'map4/map4_day_sunny_moving',
#   'map10/map10_night_both',
#   'map6/map6_night_sunny_moving',
#   'map2/map2_7583967482_stop',
#   'map6/map6_night_heavyrain_moving',
#   'map1/map1_day_sunny_moving',
#   'map1/map1_night_both',
#   'map2/map2_night_heavyrain_moving',
#   'map10/map10_day_rain_moving',
#   'map1/map1_night_sunny_moving',
#   'map5/map5_day_rain_moving',
#   'map2/map2_night_both',
#   'map3/map3_126262_both',
#   'map10/map10_day_heavyrain_moving',
#   'map2/map2_126262_stop',
#   'map5/map5_night_stop',
#   'map6/map6_126262_stop',
#   'map2/map2_day_heavyrain_moving',
#   'map1/map1_7583967482_both',
#   'map2/map2_night_rain_moving',
#   'map6/map6_day_sunny_moving',
#   'map7/map7_day_sunny_moving',
#   'map5/map5_night_sunny_moving',
#   'map7/map7_7583967482_both',
#   'map10/map10_7583967482_both',
#   'map6/map6_7583967482_both',
#   'map10/map10_night_sunny_moving',
#   'map6/map6_night_both',
#   'map4/map4_day_rain_moving',
#   'map3/map3_day_sunny_moving',
#   'map4/map4_night_both',
#   'map3/map3_day_rain_moving',
#   'map7/map7_7583967482_stop',
#   'map3/map3_night_heavyrain_moving',
#   'map5/map5_night_rain_moving',
#   'map10/map10_126262_both',
#   'map5/map5_7583967482_both',
#   'map4/map4_night_stop',
#   'map5/map5_night_heavyrain_moving',
#   'map7/map7_night_stop',
#   'map1/map1_night_heavyrain_moving',
#   'map6/map6_night_stop',
#   'map1/map1_night_rain_moving',
#   'map6/map6_126262_both',
#   'map7/map7_day_rain_moving',
#   'map10/map10_126262_stop',
#   'map7/map7_126262_both',
#   'map2/map2_day_sunny_moving',
#   'map3/map3_7583967482_stop',
#   'map3/map3_night_rain_moving',
#   'map4/map4_7583967482_stop',
#   'map5/map5_126262_both',
#   'map4/map4_night_rain_moving',
#   'map6/map6_night_rain_moving',
#   'map2/map2_day_rain_moving',
#   'map7/map7_night_both',
#   'map2/map2_night_stop',
#   'map10/map10_day_sunny_moving'],
#  'validation': ['map1/map1_day_heavyrain_moving',
#   'map5/map5_day_heavyrain_moving',
#   'map5/map5_126262_stop',
#   'map3/map3_126262_stop',
#   'map4/map4_night_sunny_moving',
#   'map5/map5_day_sunny_moving',
#   'map10/map10_night_rain_moving',
#   'map1/map1_day_rain_moving',
#   'map10/map10_7583967482_stop',
#   'map2/map2_7583967482_both',
#   'map3/map3_day_heavyrain_moving',
#   'map1/map1_7583967482_stop',
#   'map7/map7_night_heavyrain_moving',
#   'map10/map10_night_stop',
#   'map7/map7_day_heavyrain_moving',
#   'map1/map1_night_stop',
#   'map2/map2_night_sunny_moving',
#   'map10/map10_night_heavyrain_moving'],
#  'trainval': ['map6/map6_7583967482_stop',
#   'map6/map6_day_heavyrain_moving',
#   'map4/map4_night_heavyrain_moving',
#   'map7/map7_126262_stop',
#   'map5/map5_7583967482_stop',
#   'map1/map1_126262_both',
#   'map3/map3_7583967482_both',
#   'map4/map4_day_sunny_moving',
#   'map10/map10_night_both',
#   'map6/map6_night_sunny_moving',
#   'map2/map2_7583967482_stop',
#   'map6/map6_night_heavyrain_moving',
#   'map1/map1_day_sunny_moving',
#   'map1/map1_night_both',
#   'map2/map2_night_heavyrain_moving',
#   'map10/map10_day_rain_moving',
#   'map1/map1_night_sunny_moving',
#   'map5/map5_day_rain_moving',
#   'map2/map2_night_both',
#   'map3/map3_126262_both',
#   'map10/map10_day_heavyrain_moving',
#   'map2/map2_126262_stop',
#   'map5/map5_night_stop',
#   'map6/map6_126262_stop',
#   'map2/map2_day_heavyrain_moving',
#   'map1/map1_7583967482_both',
#   'map2/map2_night_rain_moving',
#   'map6/map6_day_sunny_moving',
#   'map7/map7_day_sunny_moving',
#   'map5/map5_night_sunny_moving',
#   'map7/map7_7583967482_both',
#   'map10/map10_7583967482_both',
#   'map6/map6_7583967482_both',
#   'map10/map10_night_sunny_moving',
#   'map6/map6_night_both',
#   'map4/map4_day_rain_moving',
#   'map3/map3_day_sunny_moving',
#   'map4/map4_night_both',
#   'map3/map3_day_rain_moving',
#   'map7/map7_7583967482_stop',
#   'map3/map3_night_heavyrain_moving',
#   'map5/map5_night_rain_moving',
#   'map10/map10_126262_both',
#   'map5/map5_7583967482_both',
#   'map4/map4_night_stop',
#   'map5/map5_night_heavyrain_moving',
#   'map7/map7_night_stop',
#   'map1/map1_night_heavyrain_moving',
#   'map6/map6_night_stop',
#   'map1/map1_night_rain_moving',
#   'map6/map6_126262_both',
#   'map7/map7_day_rain_moving',
#   'map10/map10_126262_stop',
#   'map7/map7_126262_both',
#   'map2/map2_day_sunny_moving',
#   'map3/map3_7583967482_stop',
#   'map3/map3_night_rain_moving',
#   'map4/map4_7583967482_stop',
#   'map5/map5_126262_both',
#   'map4/map4_night_rain_moving',
#   'map6/map6_night_rain_moving',
#   'map2/map2_day_rain_moving',
#   'map7/map7_night_both',
#   'map2/map2_night_stop',
#   'map10/map10_day_sunny_moving',
#   'map1/map1_day_heavyrain_moving',
#   'map5/map5_day_heavyrain_moving',
#   'map5/map5_126262_stop',
#   'map3/map3_126262_stop',
#   'map4/map4_night_sunny_moving',
#   'map5/map5_day_sunny_moving',
#   'map10/map10_night_rain_moving',
#   'map1/map1_day_rain_moving',
#   'map10/map10_7583967482_stop',
#   'map2/map2_7583967482_both',
#   'map3/map3_day_heavyrain_moving',
#   'map1/map1_7583967482_stop',
#   'map7/map7_night_heavyrain_moving',
#   'map10/map10_night_stop',
#   'map7/map7_day_heavyrain_moving',
#   'map1/map1_night_stop',
#   'map2/map2_night_sunny_moving',
#   'map10/map10_night_heavyrain_moving'],
#  'test': ['map4/map4_day_heavyrain_moving',
#   'map7/map7_night_sunny_moving',
#   'map4/map4_7583967482_both',
#   'map4/map4_126262_both',
#   'map7/map7_night_rain_moving',
#   'map6/map6_day_rain_moving',
#   'map4/map4_126262_stop',
#   'map5/map5_night_both',
#   'map1/map1_126262_stop',
#   'map2/map2_126262_both'],
#  'none': []}

# {'train': ['map7/map7_126262_stop',
#   'map3/map3_7583967482_stop',
#   'map2/map2_7583967482_stop',
#   'map2/map2_126262_stop',
#   'map4/map4_night_stop',
#   'map6/map6_126262_stop',
#   'map6/map6_night_stop',
#   'map10/map10_night_stop',
#   'map6/map6_7583967482_stop',
#   'map5/map5_126262_stop',
#   'map7/map7_night_stop',
#   'map5/map5_night_stop',
#   'map4/map4_7583967482_stop',
#   'map7/map7_7583967482_stop',
#   'map5/map5_7583967482_stop',
#   'map1/map1_7583967482_stop'],
#  'validation': ['map3/map3_126262_stop',
#   'map10/map10_7583967482_stop',
#   'map10/map10_126262_stop',
#   'map1/map1_night_stop'],
#  'trainval': ['map7/map7_126262_stop',
#   'map3/map3_7583967482_stop',
#   'map2/map2_7583967482_stop',
#   'map2/map2_126262_stop',
#   'map4/map4_night_stop',
#   'map6/map6_126262_stop',
#   'map6/map6_night_stop',
#   'map10/map10_night_stop',
#   'map6/map6_7583967482_stop',
#   'map5/map5_126262_stop',
#   'map7/map7_night_stop',
#   'map5/map5_night_stop',
#   'map4/map4_7583967482_stop',
#   'map7/map7_7583967482_stop',
#   'map5/map5_7583967482_stop',
#   'map1/map1_7583967482_stop',
#   'map3/map3_126262_stop',
#   'map10/map10_7583967482_stop',
#   'map10/map10_126262_stop',
#   'map1/map1_night_stop'],
#  'test': ['map1/map1_126262_stop',
#   'map2/map2_night_stop',
#   'map4/map4_126262_stop'],
#  'none': []}


# DATA_SPLIT = {'train': ['map10/map10_night_rain_moving',
#   'map1/map1_night_heavyrain_moving',
#   'map2/map2_night_rain_moving',
#   'map5/map5_night_heavyrain_moving',
#   'map5/map5_night_rain_moving',
#   'map10/map10_night_sunny_moving',
#   'map7/map7_night_heavyrain_moving',
#   'map3/map3_night_rain_moving',
#   'map10/map10_night_heavyrain_moving',
#   'map6/map6_night_sunny_moving',
#   'map4/map4_night_sunny_moving',
#   'map4/map4_night_heavyrain_moving',
#   'map1/map1_night_rain_moving',
#   'map5/map5_night_sunny_moving',
#   'map3/map3_night_heavyrain_moving',
#   'map1/map1_night_sunny_moving'],
#  'validation': ['map6/map6_night_rain_moving',
#   'map2/map2_night_sunny_moving',
#   'map2/map2_night_heavyrain_moving',
#   'map7/map7_night_sunny_moving'],
#  'trainval': ['map10/map10_night_rain_moving',
#   'map1/map1_night_heavyrain_moving',
#   'map2/map2_night_rain_moving',
#   'map5/map5_night_heavyrain_moving',
#   'map5/map5_night_rain_moving',
#   'map10/map10_night_sunny_moving',
#   'map7/map7_night_heavyrain_moving',
#   'map3/map3_night_rain_moving',
#   'map10/map10_night_heavyrain_moving',
#   'map6/map6_night_sunny_moving',
#   'map4/map4_night_sunny_moving',
#   'map4/map4_night_heavyrain_moving',
#   'map1/map1_night_rain_moving',
#   'map5/map5_night_sunny_moving',
#   'map3/map3_night_heavyrain_moving',
#   'map1/map1_night_sunny_moving',
#   'map6/map6_night_rain_moving',
#   'map2/map2_night_sunny_moving',
#   'map2/map2_night_heavyrain_moving',
#   'map7/map7_night_sunny_moving'],
#  'test': ['map6/map6_night_heavyrain_moving',
#   'map7/map7_night_rain_moving',
#   'map4/map4_night_rain_moving'],
#  'none': []}

# DATA_SPLIT = {'train': ['map3/map3_day_heavyrain_moving',
#   'map10/map10_day_sunny_moving',
#   'map7/map7_day_rain_moving',
#   'map1/map1_day_rain_moving',
#   'map10/map10_day_heavyrain_moving',
#   'map6/map6_day_sunny_moving',
#   'map5/map5_day_heavyrain_moving',
#   'map7/map7_day_heavyrain_moving',
#   'map1/map1_day_heavyrain_moving',
#   'map4/map4_day_sunny_moving',
#   'map10/map10_day_rain_moving',
#   'map3/map3_day_rain_moving',
#   'map4/map4_day_heavyrain_moving',
#   'map7/map7_day_sunny_moving',
#   'map5/map5_day_sunny_moving',
#   'map2/map2_day_sunny_moving'],
#  'validation': ['map1/map1_day_sunny_moving',
#   'map2/map2_day_rain_moving',
#   'map3/map3_day_sunny_moving',
#   'map6/map6_day_rain_moving'],
#  'trainval': ['map3/map3_day_heavyrain_moving',
#   'map10/map10_day_sunny_moving',
#   'map7/map7_day_rain_moving',
#   'map1/map1_day_rain_moving',
#   'map10/map10_day_heavyrain_moving',
#   'map6/map6_day_sunny_moving',
#   'map5/map5_day_heavyrain_moving',
#   'map7/map7_day_heavyrain_moving',
#   'map1/map1_day_heavyrain_moving',
#   'map4/map4_day_sunny_moving',
#   'map10/map10_day_rain_moving',
#   'map3/map3_day_rain_moving',
#   'map4/map4_day_heavyrain_moving',
#   'map7/map7_day_sunny_moving',
#   'map5/map5_day_sunny_moving',
#   'map2/map2_day_sunny_moving',
#   'map1/map1_day_sunny_moving',
#   'map2/map2_day_rain_moving',
#   'map3/map3_day_sunny_moving',
#   'map6/map6_day_rain_moving'],
#  'test': ['map2/map2_day_heavyrain_moving',
#   'map5/map5_day_rain_moving',
#   'map4/map4_day_rain_moving',
#   'map6/map6_day_heavyrain_moving'],
#  'none': []}

# 'map5_no_bf','map7_no_bf'
# 'map5_no_bf','map7_no_bf'
# 'map5_no_bf','map7_no_bf'
