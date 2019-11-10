
import cv2
import time
import tracking
import numpy as np
import tensorflow as tf
from config import FLAGS
from keras.models import load_model
from model.cpm_model import CPM_Model
from model.keypoint_detector import *
from keras.models import model_from_json


graph1 = tf.Graph()
with graph1.as_default():
    session1 = tf.Session()
    with session1.as_default():
        with open('models/gesture/model.json') as arch_file:
            ahm = model_from_json(arch_file.read())
        ahm.load_weights('models/gesture/model.h5')


def main(argv):
    joint_detections = np.zeros(shape=(21, 2))

    tracker = tracking.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)

    model = CPM_Model(input_size=FLAGS.input_size,
                                heatmap_size=FLAGS.heatmap_size,
                                stages=FLAGS.cpm_stages,
                                joints=FLAGS.num_of_joints,
                                img_type=FLAGS.color_channel,
                                is_training=False)
    saver = tf.train.Saver()

    output_node = tf.get_default_graph().get_tensor_by_name(name=FLAGS.output_node_names)
    FLAGS.use_gpu = 1
    device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
    sess_config = tf.ConfigProto(device_count=device_count)

    with tf.Session(config=sess_config) as sess:

        saver.restore(sess, 'models/keypoint/cpm_hand')

        cam = cv2.VideoCapture(0)

        kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
        for _, joint_kalman_filter in enumerate(kalman_filter_array):
            joint_kalman_filter.transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                np.float32)
            joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                           np.float32) * FLAGS.kalman_noise

        while True:
            _, full_img = cam.read()

            test_img = tracker.tracking_by_joints(full_img, joint_detections=joint_detections)
            crop_full_scale = tracker.input_crop_ratio
            test_img_copy = test_img.copy()

            test_img_input = normalize_and_centralize_img(test_img)
            t1 = time.time()
            stage_heatmap_np = sess.run([output_node],
                                        feed_dict={model.input_images: test_img_input})

            print('FPS: %.2f' % (1 / (time.time() - t1)))
            local_img, joint_detections = visualize_result(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale,
                                         test_img_copy, joint_detections)
            cv2.imshow('global_img', full_img.astype(np.uint8))
            if cv2.waitKey(1) == ord('q'): print("To abort: Ctrl + C")



if __name__ == '__main__':
    tf.app.run()
    
