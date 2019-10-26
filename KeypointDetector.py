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







#____________________________________________________________________________________________________________________________________________________________________________________________________________________________
from tkinter import *
from tkinter import scrolledtext
from action_interface import *
import time


window = Tk()

window.title("Tele Mouse")
window.geometry('350x275')


#lbl = Label(window, text="Click to start!", font = ('Comic Sans MS',25))
#lbl.grid(column=0, row=0)


def clicked():

    btn.configure(text="Button was clicked !!")


btn = Button(window, text="Click Me", command = clicked)

btn.grid(column=0, row=5)

def leftKey(event):
    myActor.test_action('10000000')
    return

def rightKey(event):
    myActor.test_action('01000000')
    return

def upKey(event):
    myActor.test_action('00100000')
    return

def downKey(event):
    myActor.test_action('00010000')
    return

myActor = Actor()

frame = Frame(window, width=100, height=100)
window.bind('<Left>', leftKey)
window.bind('<Right>', rightKey)
window.bind('<Up>', upKey)
window.bind('<Down>', downKey)
frame.grid()

txt = scrolledtext.ScrolledText(window,width=40,height=10)
txt.grid(column=0,row=0)

txt.config(state=DISABLED)


#____________________________________________________________________________________________________________________________________________________________________________________________________________________________





















#ahm = load_model('my_model.h5')


graph1 = tf.Graph()
with graph1.as_default():
	session1 = tf.Session()
	with session1.as_default():
		with open('model.json') as arch_file:
			ahm = model_from_json(arch_file.read())
		ahm.load_weights('model.h5')

joint_detections = np.zeros(shape=(21, 2))

#ahm = model_from_json(loaded_model_json)
#ahm.load_weights("model.h5")

global myvar3
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

			loopfunc(ahm,model,output_node,cam,kalman_filter_array,tracker,sess,joint_detections)
































def loopfunc(ahm,model,output_node,cam,kalman_filter_array,tracker,sess,joint_detections):
			_, full_img = cam.read()

			test_img = tracker.tracking_by_joints(full_img, joint_detections=joint_detections)
			crop_full_scale = tracker.input_crop_ratio
			test_img_copy = test_img.copy()

			test_img_input = normalize_and_centralize_img(test_img)


			t1 = time.time()
			stage_heatmap_np = sess.run([output_node],
										feed_dict={model.input_images: test_img_input})

			print('FPS: %.2f' % (1 / (time.time() - t1)))
			#for item in joint_detections:
			#	print(item)

			local_img = visualize_result(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale,
										 test_img_copy)


			j2 = np.reshape(joint_detections,(1,42))
			j2= np.append(j2,0)
			j2= np.append(j2,0)
			j2 = np.reshape(j2,(1,44))


			try:
				if(j2[0,19]>430):
					myActor.test_action('10000000')
				elif(j2[0,19]<120):
					myActor.test_action('01000000')
				if(j2[0,18]>380):
					myActor.test_action('00010000')
				if(j2[0,18]<120):
					myActor.test_action('00100000')
			except:
				print('not passed yet')

			#print(j2.shape)

			with graph1.as_default():
				
				with session1.as_default():
					print(j2)
					a=ahm.predict(j2)
			print(a)
			myvar3=str(list(((a.max(axis=1,keepdims=1) == a) + [0,0,0,0,0,0,0,0])[0]))
			print(np.shape(myvar3),type(myvar3))
			print(myvar3)
			myActor.act(myvar3)
			cv2.imshow('global_img', full_img.astype(np.uint8))
			if cv2.waitKey(1) == ord('q'): print("To abort: Ctrl + C")
			window.update_idletasks()
			window.update()
















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

	#for item in joint_coord_set:
	#	print(item)

	draw_hand(full_img, joint_coord_set, tracker.loss_track)
	draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
	joint_detections = joint_coord_set

	if mean_response_val >= 1:
		tracker.loss_track = False
	else:
		tracker.loss_track = True

	cv2.putText(full_img, 'Response: {:<.3f}'.format(mean_response_val),
				org=(20, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0))


def draw_hand(full_img, joint_coords, is_loss_track):
	if is_loss_track:
		joint_coords = FLAGS.default_hand

	
	for joint_num in range(FLAGS.num_of_joints):
		#color_code_num = (joint_num // 4)
		#radiN = 5
		#jointS = 255 - (joint_num%5)*50
		cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=4, color=(255,255,255), thickness=-1)	
		'''
		if joint_num in [0]:
			joint_color = (jointS,0,0)
			cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=7,
					   color=joint_color, thickness=-1)	
		elif joint_num in [0,1,2,3,4]:
			joint_color = (jointS,0,0)
			cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=radiN,
					   color=joint_color, thickness=-1)		
		elif joint_num in [5,6,7,8,9]:
			joint_color = (0,jointS,0)
			cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=radiN,
					   color=joint_color, thickness=-1)
		elif joint_num in [10,11,12,13,14]:
			joint_color = (0,0,jointS)
			cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=radiN,
					   color=joint_color, thickness=-1)
		else:
			joint_color = (jointS,jointS,jointS)
			cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=radiN,
					   color=joint_color, thickness=-1)
		'''	
	'''
		elif joint_num in [4, 8, 12, 16]:
			#joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
			cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
					   color=joint_color, thickness=-1)
		else:
			#joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
			cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
					   color=joint_color, thickness=-1)
	'''

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
	
