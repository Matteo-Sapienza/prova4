from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
from mtcnn_cv2 import MTCNN

import torch, face_detection
from models import Wav2Lip
import platform

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)

parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, help='If True, then use only first video frame for inference', default=False)

parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=16)

parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.''Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true', help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.' 'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true', help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

##don't use it anymore!!!
def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				##do face detection (with the imported function)
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
		
		#create rect (obtained with face detection)
		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)

	##do smooth (if required)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)

	##save on results the images cropped (used face detector)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
	detector = MTCNN()
	##face_detect() return an array of immages cropped and coordinates with only faces (obtained with face detection)
	if args.box[0] == -1:
		if not args.static:
			#face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
			face_det_results = []
			
			cropped = []
			detector = MTCNN()
			print('\nLet\'s go')
			i = 0
			threshold = 3
			while i < len(frames):
				image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
				result = detector.detect_faces(image)
				if result != []:
					try:
						a = frames[i][result[0]['box'][1]+10:result[0]['box'][1]+result[0]['box'][3]-10, result[0]['box'][0]+10:result[0]['box'][0]+result[0]['box'][2]-10]
						c = 0
						while c < threshold:
							face_det_results.append((result[0]['box'][1]+10, result[0]['box'][1] + result[0]['box'][3]-10, result[0]['box'][0]+10, result[0]['box'][0] + result[0]['box'][2]-10))
							cropped.append(a)
							c += 1
					except:
						a = frames[i][result[0]['box'][1]:result[0]['box'][1]+result[0]['box'][3], result[0]['box'][0]:result[0]['box'][0]+result[0]['box'][2]]
						c = 0
						while c < threshold:
							face_det_results.append((result[0]['box'][1], result[0]['box'][1] + result[0]['box'][3], result[0]['box'][0], result[0]['box'][0] + result[0]['box'][2]))
							cropped.append(a)
							c += 1
					#cropped.append(image[x:x+w, y:y+h])
				else:
					c = 0
					while c < threshold:
						face_det_results.append(face_det_results[-1])
						cropped.append(cropped[-1])
						c += 1
				if i + 3 > len(frames):
					threshold = len(frames) - i
					i += len(frames) - 1
				else:
					i += 3
		else:
			#face_det_results = face_detect([frames[0]])
			image = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
			result = detector.detect_faces(image)
			face_det_results.append([result[0]['box'][1], result[0]['box'][1] + result[0]['box'][3], result[0]['box'][0], result[0]['box'][0] + result[0]['box'][2]])
			a = frames[0][result[0]['box'][1]:result[0]['box'][1]+result[0]['box'][3], result[0]['box'][0]:result[0]['box'][0]+result[0]['box'][2]]
			cropped.append(a)

	##create an array of immages cropped and coordinates with only faces using the args passed by the user
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		for f in frames:
			face_det_results.append([y1, y2, x1, x2])
			a = frames[i][y1: y2, x1:x2]
			cropped.append(a)

	

	##process and adapted frames with audio file
	print('\nOk')
	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy() ##original frame
		face = cropped[idx].copy()
		coords = face_det_results[idx]
		#face, coords = face_det_results[idx].copy() ##cropped frame + coords

		face = cv2.resize(face, (args.img_size, args.img_size)) ##args.img_size = 96 (by default)  [why?]
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)
		
		##reshape length to be conformed to batch_size(?)
		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch) ##np.asarray => transform input arg in to array

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch ##returns the generator for these lists
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch ##returns the generator for these lists

##defining variable and device
mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

##function to calcolate time set
if device == 'cuda':
	torch.cuda.synchronize()
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
print('Using {} for inference.'.format(device))

##load and return checkpoint for model Wav2Lip
def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	return checkpoint

##load and return model Wav2Lip (using pretrained weights)
def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def main():
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps

	else:
		##process video (dividing it in frames)
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			##in case of resizing
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
			
			##in case of rotating
			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
			
			##default doesn't crop
			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	print ("Number of frames available for inference: "+str(len(full_frames)))

	##process audio (converting if not .wav)
	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'
	
	##Load audio
	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	##np.isnan => return true if inside there is not a number
	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
	
	##process audio
	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	##adapt video to the length of audio
	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size

	##start calcolate time
	start.record()


	gen = datagen(full_frames.copy(), mel_chunks) ##returns the generator for lists: img_batch(faces), mel_batch(audio), frame_batch(original frames), coords_batch(coords of faces)

	##np.ceil(a) => return all the elemnt of 'a' list, rounded on top (es 0.1 => 1)
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):

		##only first iteration
		if i == 0:
			model = load_model(args.checkpoint_path)
			print ("Model loaded")


			##set height and weight
			frame_h, frame_w = full_frames[0].shape[:-1]
			#save video in 'temp/result.avi'
			out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))


		##[why transpose?] (maybe for the model)
		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		##[why contro-transpose?] (maybe decode the output of the model)
		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c ##unzip coords
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1)) ##convert tensor p in np.uint8 and then resize

			f[y1:y2, x1:x2] = p ##overwrite pred on the original frame
			out.write(f) ##save all on the temp file ('temp/result.avi')

	out.release()
	

	#save the temp file ('temp/result.avi') on the output path (default: 'results/result_voice.mp4')
	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')
	
	#end calcolate time
	end.record()
	torch.cuda.synchronize()
	print('Time: ', start.elapsed_time(end))

if __name__ == '__main__':
	main()
