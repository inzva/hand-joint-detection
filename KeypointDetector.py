import tensorflow as tf
import numpy as np
import tracking_module
import cv2
import time
import math
from model import CPM_Model
from config import FLAGS
from keras.models import load_model
from keras.models import model_from_json
import pyautogui as mouse
import time

mouse.FAILSAFE = False


limiter =1
counter = 0
control_distance = 55
mov_speed = 0.25
last_click_time = time.time()



graph1 = tf.Graph()
with graph1.as_default():
	session1 = tf.Session()
	with session1.as_default():
		with open('model.json') as arch_file:
			ahm = model_from_json(arch_file.read())
		ahm.load_weights('model.h5')




joint_detections = np.zeros(shape=(21, 2))














def main(argv):



	tracker = tracking_module.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)


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

		saver.restore(sess, 'model/cpm_hand')

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




			local_img = visualize_result(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale,
										 test_img_copy)



			cv2.imshow('global_img', full_img.astype(np.uint8))
			if cv2.waitKey(1) == ord('q'): print("To abort: Ctrl + C")













def normalize_and_centralize_img(img):

	test_img_input = img / 255
	test_img_input = np.expand_dims(test_img_input, axis=0)

	return test_img_input






def visualize_result(test_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
	demo_stage_heatmaps = []


	last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
		(FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
	last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))


	correct_and_draw_hand(test_img, last_heatmap, kalman_filter_array, tracker, crop_full_scale, crop_img)

	return crop_img






def correct_and_draw_hand(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
	global joint_detections
	joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))
	local_joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))

	mean_response_val = 0.0



	for joint_num in range(FLAGS.num_of_joints):
		tmp_heatmap = stage_heatmap_np[:, :, joint_num]
		joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
									   (FLAGS.input_size, FLAGS.input_size))
		mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
		joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
		kalman_filter_array[joint_num].correct(joint_coord)
		kalman_pred = kalman_filter_array[joint_num].predict()
		correct_coord = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
		local_joint_coord_set[joint_num, :] = correct_coord

		correct_coord /= crop_full_scale

		correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
		correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
		correct_coord[0] += tracker.bbox[0]
		correct_coord[1] += tracker.bbox[2]
		joint_coord_set[joint_num, :] = correct_coord



	draw_hand(full_img, joint_coord_set, tracker.loss_track)
	#draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
	joint_detections = joint_coord_set

	if mean_response_val >= 1:
		tracker.loss_track = False
	else:
		tracker.loss_track = True

	cv2.putText(full_img, 'Response: {:<.3f}'.format(mean_response_val),
				org=(20, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0))










def draw_hand(full_img, joint_coords, is_loss_track):

	global control_distance	
	global mov_speed




	if is_loss_track:
		joint_coords = FLAGS.default_hand

	
	for joint_num in range(FLAGS.num_of_joints):


		if(joint_num!=9):
			cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=4, color=(255,255,255), thickness=-1)

	
		else:
			#We get inside this else only for our control joint.
	
			#reference point
			cv2.circle(full_img, center=(328, 199), radius=5, color=(0,255,0), thickness=-1)

			#controlling point
			cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=4, color=(0,0,255), thickness=-1)	

			#reference circle
			cv2.circle(full_img, center=(328, 199), radius=control_distance, color=(0,255,0), thickness=3)	
			#print('COORD',int(joint_coords[joint_num][0]),int(joint_coords[joint_num][1]))



			rect_hor_len = 200	
			rect_ver_len = 100

			rect_left = int(328 - (rect_hor_len/2))
			rect_right = rect_left + rect_hor_len


			rect_up = 199 + control_distance + 50
			rect_down = rect_up + rect_ver_len
			
			#print(rect_up,rect_down,rect_left,rect_right)
			
		

			cv2.line(full_img, (rect_left,rect_up), (rect_right,rect_up), color = (0,0,255) ) 
			cv2.line(full_img, (rect_left,rect_down), (rect_right,rect_down), color = (0,0,255) )
			cv2.line(full_img, (rect_left,rect_down), (rect_left,rect_up), color = (0,0,255) )
			cv2.line(full_img, (rect_right,rect_down), (rect_right,rect_up), color = (0,0,255) )



			distance = math.sqrt((joint_coords[joint_num][0]-199)**2 + (joint_coords[joint_num][1]-328)**2)
			g_dist_x = joint_coords[4][1] - joint_coords[20][1]
			print('GDIST',g_dist_x)

			global last_click_time
			time_passed = time.time() - last_click_time
			if(g_dist_x<0 and time_passed>1.5):
				mouse.click()
				last_click_time = time.time()

			if(distance > control_distance and g_dist_x>100):
				#We get in here only if we are outside the reference circle

				if(joint_coords[joint_num][0]<rect_up):
					#print('RECT UP VS CONT POINT',joint_coords[joint_num][1],rect_up)
					#print('Dot is OUTSIDE. Distance: {0}'.format(distance))



					global counter, limiter

					if(counter==limiter):
						counter = 0

						mydif = (joint_coords[joint_num][1]-328,joint_coords[joint_num][0]-199)
						mymov = ( mydif[0] * mov_speed , mydif[1] * mov_speed)

						mouse.moveRel(mymov[0],mymov[1], duration=0.05)


					else:
						counter = counter + 1
				
				else:
					control_point = (int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0]))
					loc_ratio = ((control_point[0]-rect_left)/rect_hor_len , (control_point[1]-rect_up)/rect_ver_len)
					myloc = (int(loc_ratio[0]*1920),int(loc_ratio[1]*1080))
					print('Second Mode Active. MYLOC: ',myloc[0],myloc[1])
					g_dist_x = joint_coords[0][0] - joint_coords[20][0]
					print('GDISTX',g_dist_x)
					if(joint_coords[joint_num][1]<rect_down and joint_coords[joint_num][0]>rect_left and joint_coords[joint_num][0]<rect_right and g_dist_x>100 ):
						mouse.moveTo(myloc[0],myloc[1])

					else:
						continue


			else:
				#print('Dot is INSIDE. Distance: {0}'.format(distance))
				continue



	draw_limbs(full_img,joint_coords)

















def draw_limbs(full_img,joint_coords):

	for limb_num in range(len(FLAGS.limbs)):
		x1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][0])
		y1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][1])
		x2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][0])
		y2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][1])
		length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
		if length < 150 and length > 5:
			deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
			polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
									   (int(length / 2), 3),
									   int(deg),
									   0, 360, 1)
			color_code_num = limb_num // 4
			limb_color = list(map(lambda x: x + 35 * (limb_num % 4), FLAGS.joint_color_code[color_code_num]))
			cv2.fillConvexPoly(full_img, polygon, color=limb_color)




if __name__ == '__main__':
	tf.app.run()
	
