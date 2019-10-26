class FLAGS(object):

    input_size = 256 
    heatmap_size = 32
    cpm_stages = 3
    num_of_joints = 21
    color_channel = 'RGB'
    normalize_img = True
    use_gpu = False
    gpu_id = 0

    webcam_height = 480
    webcam_width = 640

    use_kalman = True
    kalman_noise = 0.03


    output_node_names = 'stage_3/mid_conv7/BiasAdd:0'

    default_hand = [[259, 335],
                    [245, 311],
                    [226, 288],
                    [206, 270],
                    [195, 261],
                    [203, 308],
                    [165, 290],
                    [139, 287],
                    [119, 284],
                    [199, 328],
                    [156, 318],
                    [128, 314],
                    [104, 318],
                    [204, 341],
                    [163, 340],
                    [133, 347],
                    [108, 349],
                    [206, 359],
                    [176, 368],
                    [164, 370],
                    [144, 377]]


    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    joint_color_code = [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]
