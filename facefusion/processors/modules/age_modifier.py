from argparse import ArgumentParser
from typing import Any, List
import numpy as np
import cv2
from cv2.typing import Size
from numpy.typing import NDArray
from PIL import Image
import time

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
import facefusion.processors.core as processors
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, process_manager, state_manager, wording
from facefusion.common_helper import create_int_metavar
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_helper import merge_matrix, paste_back, scale_face_landmark_5, warp_face_by_face_landmark_5
from facefusion.face_masker import create_occlusion_mask, create_static_box_mask
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import in_directory, is_image, is_video, is_file, resolve_relative_path, same_file_extension
from facefusion.processors import choices as processors_choices
from facefusion.processors.typing import AgeModifierInputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import thread_semaphore
from facefusion.typing import ApplyStateItem, Args, Face, InferencePool, Mask, ModelOptions, ModelSet, ProcessMode, QueuePayload, UpdateProgress, VisionFrame
from facefusion.vision import read_image, read_static_image, write_image


MODEL_SET : ModelSet =\
{
	'fran':
	{
		'hashes':
		{
			'age_modifier':
			{
				'url': 'https://github.com/TMASynthetics/facefusion3/releases/download/v3.0.1/fran_onnx.hash',
				'path': resolve_relative_path("../.assets/models/fran_onnx.hash"),
			}
		},
		'sources':
		{
			'age_modifier':
			{
				'url': 'https://github.com/TMASynthetics/facefusion3/releases/download/v3.0.1/fran.onnx',
				'path': resolve_relative_path("../.assets/models/fran.onnx"),
			},
		},	
		'masks':
        {
            'mask':
			{	
				'url': 'https://github.com/TMASynthetics/facefusion3/releases/download/v3.0.1/mask1024.jpg',
				'path': resolve_relative_path("../.assets/mask1024.jpg"),
			},
            'small_mask':
			{
				#'url': 'https://github.com/TMASynthetics/facefusion3/releases/download/v3.0.1/mask512.jpg',
				'url': 'https://github.com/Schumix60/models/releases/download/weights/mask512.jpg',
				'path': resolve_relative_path("../.assets/mask512.jpg"),
			}
		},
		'template': 'ffhq_512',
		'window_size': 512,
		'size': (1024, 1024)
	},
    
	'styleganex_age':
	{
		'hashes':
		{
			'age_modifier':
			{
				'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/styleganex_age.hash',
				'path': resolve_relative_path('../.assets/models/styleganex_age.hash')
			}
		},
		'sources':
		{
			'age_modifier':
			{
				'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/styleganex_age.onnx',
				'path': resolve_relative_path('../.assets/models/styleganex_age.onnx')
			}
		},
		'template': 'ffhq_512',
		'size': (512, 512)
	}
}


def get_inference_pool() -> InferencePool:
	model_sources = get_model_options().get('sources')
	model_context = __name__ + '.' + state_manager.get_item('age_modifier_model')
	return inference_manager.get_inference_pool(model_context, model_sources)


def clear_inference_pool() -> None:
	model_context = __name__ + '.' + state_manager.get_item('age_modifier_model')
	inference_manager.clear_inference_pool(model_context)


def get_model_options() -> ModelOptions:
	age_modifier_model = state_manager.get_item('age_modifier_model')
	return MODEL_SET.get(age_modifier_model)


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--age-modifier-model', help = wording.get('help.age_modifier_model'), default = config.get_str_value('processors.age_modifier_model', 'fran'), choices = processors_choices.age_modifier_models)
		group_processors.add_argument('--age-modifier-direction', help = wording.get('help.age_modifier_direction'), type = int, default = config.get_int_value('processors.age_modifier_direction', '0'), choices = processors_choices.age_modifier_direction_range, metavar = create_int_metavar(processors_choices.age_modifier_direction_range))
		group_processors.add_argument('--age-modifier-source-age', help = wording.get('help.age_modifier_source_age'), type = int, default = config.get_int_value('processors.age_modifier_source_age', '20'), choices = processors_choices.age_modifier_source_age_range, metavar = create_int_metavar(processors_choices.age_modifier_source_age_range))
		group_processors.add_argument('--age-modifier-target-age', help = wording.get('help.age_modifier_target_age'), type = int, default = config.get_int_value('processors.age_modifier_target_age', '70'), choices = processors_choices.age_modifier_target_age_range, metavar = create_int_metavar(processors_choices.age_modifier_target_age_range))
		group_processors.add_argument('--age-modifier-stride', help = wording.get('help.age_modifier_stride'), type = int, default = 256) 
		group_processors.add_argument('--age-modifier-show-mask', type = str, default = "No") 

		facefusion.jobs.job_store.register_step_keys([ 'age_modifier_model', 'age_modifier_direction', 'age_modifier_source_age','age_modifier_target_age', 'age_modifier_stride', 'age_modifier_show_mask' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('age_modifier_model', args.get('age_modifier_model'))
	apply_state_item('age_modifier_direction', args.get('age_modifier_direction'))
	apply_state_item('age_modifier_source_age', args.get('age_modifier_source_age'))
	apply_state_item('age_modifier_target_age', args.get('age_modifier_target_age'))
	apply_state_item('age_modifier_stride', args.get('age_modifier_stride'))
	apply_state_item('age_modifier_show_mask', args.get('age_modifier_show_mask'))


def pre_check() -> bool:
	download_directory_path = resolve_relative_path('../.assets/models')
	model_hashes = get_model_options().get('hashes')
	model_sources = get_model_options().get('sources')

	age_modifier_model = state_manager.get_item('age_modifier_model')

	# manage the manual download of fran assets
	if age_modifier_model == 'fran':
		model_path = model_sources.get('age_modifier').get('path')
		model_url = model_sources.get('age_modifier').get('url')
		mask_path = get_model_options().get('masks').get("mask").get("path")
		mask_url = get_model_options().get('masks').get("mask").get("url")
		small_mask_path = get_model_options().get('masks').get("small_mask").get("path")
		small_mask_url = get_model_options().get('masks')
		if not is_file(model_path):
			#return conditional_download_sources(download_directory_path, model_sources)
			logger.error(wording.get('help.download_fran_model_first') + wording.get('exclamation_mark') + ' : ' + model_url, __name__)
			return False
		if not is_file(mask_path):
			logger.error(wording.get('help.download_fran_masks_first') + wording.get('exclamation_mark') + ' : ' + mask_url, __name__)
			return False
		if not is_file(small_mask_path):			
			logger.error(wording.get('help.download_fran_masks_first') + wording.get('exclamation_mark') + ' : ' + small_mask_url, __name__)
			return False
		return True
	else:
		return conditional_download_hashes(download_directory_path, model_hashes) and conditional_download_sources(download_directory_path, model_sources)
	

def pre_process(mode : ProcessMode) -> bool:
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(wording.get('choose_image_or_video_target') + wording.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(wording.get('specify_image_or_video_output') + wording.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension([ state_manager.get_item('target_path'), state_manager.get_item('output_path') ]):
		logger.error(wording.get('match_target_and_output_extension') + wording.get('exclamation_mark'), __name__)
		return False
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		face_classifier.clear_inference_pool()
		face_detector.clear_inference_pool()
		face_landmarker.clear_inference_pool()
		face_masker.clear_inference_pool()
		face_recognizer.clear_inference_pool()


def apply_fran_re_aging(input_array, window_size, stride, mask_array, small_mask_array):
	"""
	Optimized version to apply aging operation using a sliding-window method with an ONNX model, using NumPy arrays.
	"""
	print('apply_fran_re_aging')
	start_total = time.time()
	age_modifier = get_inference_pool().get("age_modifier")

	n, c, h, w = input_array.shape
	output_array = np.zeros((n, 3, h, w), dtype=input_array.dtype)
	count_array = np.zeros((n, 3, h, w), dtype=np.float32)
	add = 2 if window_size % stride != 0 else 1

	for y in range(0, h - window_size + add, stride):
		for x in range(0, w - window_size + add, stride):
			window = input_array[:, :, y:y + window_size, x:x + window_size]

			# ONNX inference
			age_modifier_inputs = {'input': window}
			with thread_semaphore():
				output_onnx = age_modifier.run(None, age_modifier_inputs)[0]

			output_array[:, :, y:y + window_size, x:x + window_size] += output_onnx * small_mask_array
			count_array[:, :, y:y + window_size, x:x + window_size] += small_mask_array

	count_array = np.clip(count_array, a_min=1.0, a_max=None)	
	output_array /= count_array # Average the overlapping regions
	output_array *= mask_array # Apply mask
	print('TOTAL inference time (s) : ', time.time() - start_total)
	return output_array


def modify_age(target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	age_modifier_model = state_manager.get_item('age_modifier_model')
	if age_modifier_model == 'fran':
		# Load model options and masks
		mask_path = get_model_options().get('masks').get("mask").get("path")
		small_mask_path = get_model_options().get('masks').get("small_mask").get("path")
		input_size = get_model_options().get('size')  # (1024, 1024)
		window_size = get_model_options().get('window_size') # 512 
		stride = state_manager.get_item('age_modifier_stride')

		# Load and normalize masks using NumPy
		mask_array = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255
		small_mask_array = np.array(Image.open(small_mask_path).convert('L'), dtype=np.float32) / 255

		# Load and preprocess image
		image = temp_vision_frame.copy() # (H, W, C)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255  # (H, W, C) Normalize to [0, 1]

		# calculate margins
		x1, y1, x2, y2 = target_face.bounding_box
		
		face_mask_padding = state_manager.get_item("face_mask_padding") # top, right, bottom, left
		x1 += face_mask_padding[3]
		y1 += face_mask_padding[0]
		x2 += face_mask_padding[1]
		y2 += face_mask_padding[2]

		margin_y_t = int((y2 - y1) * .63 * .85)  # Calculate the top margin to extend above the face for better coverage.
		margin_y_b = int((y2 - y1) * .37 * .85)  # Calculate the bottom margin.
		margin_x = int((x2 - x1) // (2 / .85))  # Calculate the horizontal margin for a square crop.
		margin_y_t += 2 * margin_x - margin_y_t - margin_y_b  # Adjust the top margin to ensure a square crop.
		
		l_y = int(max([y1 - margin_y_t, 0]))  # Determine the top boundary of the crop, ensuring it doesn't go below zero.
		r_y = int(min([y2 + margin_y_b, image.shape[0]]))  # Determine the bottom boundary, ensuring it stays within the image height.
		l_x = int(max([x1 - margin_x, 0]))  # Determine the left boundary of the crop.
		r_x = int(min([x2 + margin_x, image.shape[1]]))  # Determine the right boundary.
	
		# Crop the image to the computed boundaries
		cropped_image = image[l_y:r_y, l_x:r_x, :] # (H, W, C) cropped

		# Resize it using OpenCV
		cropped_image_resized = cv2.resize(cropped_image, input_size, interpolation=cv2.INTER_LINEAR) # (H, W, C) (1024, 1024, 3)
		cropped_image_resized = np.transpose(cropped_image_resized, (2, 0, 1))  # [C, H, W] (3, 1024, 1024)

		# Prepare input array
		source_age = state_manager.get_item('age_modifier_source_age') if state_manager.get_item('age_modifier_source_age') else 20
		target_age = state_manager.get_item('age_modifier_target_age') if state_manager.get_item('age_modifier_target_age') else 80

		source_age_channel = np.full_like(cropped_image_resized[:1, :, :], source_age / 100) # create a channel for source_age (1, 1024, 1024)
		target_age_channel = np.full_like(cropped_image_resized[:1, :, :], target_age / 100) # create a channel for target_age (1, 1024, 1024)
		input_array = np.concatenate([cropped_image_resized, source_age_channel, target_age_channel], axis=0)[np.newaxis, ...] # (1, 5, 1024, 1024)

		aged_cropped_image = apply_fran_re_aging(input_array, window_size, stride, mask_array, small_mask_array) # (1, 3, 1024, 1024)

		# Resize back to original size using OpenCV
		aged_cropped_image_resized = cv2.resize(np.transpose(aged_cropped_image[0], (1, 2, 0)), (r_x - l_x, r_y - l_y), interpolation=cv2.INTER_LINEAR) # [H, W, C]

		# Reapply to original image
		image[l_y:r_y, l_x:r_x, :] += aged_cropped_image_resized # (H, W, C)
		image = np.clip(image, 0, 1) # (H, W, C) [0-1]
		
		# Convert to final output format
		paste_vision_frame = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)  # (H, W, C) [0-255]

		# show rectangle to see the cropping rectangle (after the aging process)
		if state_manager.get_item('age_modifier_show_mask') == "Yes":
			cv2.rectangle(paste_vision_frame, (l_x, l_y), (r_x, r_y), (0, 0, 255), 3)
		
	else:
		print('styleganex_age')
		model_template = get_model_options().get('template')
		model_size = get_model_options().get('size')
		crop_size = (model_size[0] // 2, model_size[1] // 2)  # divide the size by 2
		face_landmark_5 = target_face.landmark_set.get('5/68').copy()
		extend_face_landmark_5 = scale_face_landmark_5(face_landmark_5, 2.0) # scale the 4 extrem landmark (eye and mouth) with the nose as the center.
		crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, model_template, crop_size)
		extend_vision_frame, extend_affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, extend_face_landmark_5, model_template, model_size)
		extend_vision_frame_raw = extend_vision_frame.copy()
		box_mask = create_static_box_mask(model_size, state_manager.get_item('face_mask_blur'), (0, 0, 0, 0))
		
		crop_masks =\
		[
			box_mask
		]

		if 'occlusion' in state_manager.get_item('face_mask_types'):
			occlusion_mask = create_occlusion_mask(crop_vision_frame)
			combined_matrix = merge_matrix([ extend_affine_matrix, cv2.invertAffineTransform(affine_matrix) ])
			occlusion_mask = cv2.warpAffine(occlusion_mask, combined_matrix, model_size)
			crop_masks.append(occlusion_mask)

		crop_vision_frame = prepare_vision_frame(crop_vision_frame) # (w, h, c) -> (b, c, w, h)
		extend_vision_frame = prepare_vision_frame(extend_vision_frame)
		extend_vision_frame = forward(crop_vision_frame, extend_vision_frame)
		extend_vision_frame = normalize_extend_frame(extend_vision_frame)
		extend_vision_frame = fix_color(extend_vision_frame_raw, extend_vision_frame)
		extend_crop_mask = cv2.pyrUp(np.minimum.reduce(crop_masks).clip(0, 1)) # double the size of the resulting mask
		extend_affine_matrix *= extend_vision_frame.shape[0] / 512
		paste_vision_frame = paste_back(temp_vision_frame, extend_vision_frame, extend_crop_mask, extend_affine_matrix)

	return paste_vision_frame


def forward(crop_vision_frame : VisionFrame, extend_vision_frame : VisionFrame) -> VisionFrame:
	age_modifier = get_inference_pool().get('age_modifier')
	age_modifier_inputs = {}

	for age_modifier_input in age_modifier.get_inputs():
		if age_modifier_input.name == 'target':
			age_modifier_inputs[age_modifier_input.name] = crop_vision_frame
		if age_modifier_input.name == 'target_with_background':
			age_modifier_inputs[age_modifier_input.name] = extend_vision_frame
		if age_modifier_input.name == 'direction':
			age_modifier_inputs[age_modifier_input.name] = prepare_direction(state_manager.get_item('age_modifier_direction'))

	with thread_semaphore():
		crop_vision_frame = age_modifier.run(None, age_modifier_inputs)[0][0]

	return crop_vision_frame


def fix_color(extend_vision_frame_raw : VisionFrame, extend_vision_frame : VisionFrame) -> VisionFrame:
	color_difference = compute_color_difference(extend_vision_frame_raw, extend_vision_frame, (48, 48))
	color_difference_mask = create_static_box_mask(extend_vision_frame.shape[:2][::-1], 1.0, (0, 0, 0, 0))
	color_difference_mask = np.stack((color_difference_mask, ) * 3, axis = -1)
	extend_vision_frame = normalize_color_difference(color_difference, color_difference_mask, extend_vision_frame)
	return extend_vision_frame


def compute_color_difference(extend_vision_frame_raw : VisionFrame, extend_vision_frame : VisionFrame, size : Size) -> VisionFrame:
	extend_vision_frame_raw = extend_vision_frame_raw.astype(np.float32) / 255
	extend_vision_frame_raw = cv2.resize(extend_vision_frame_raw, size, interpolation = cv2.INTER_AREA)
	extend_vision_frame = extend_vision_frame.astype(np.float32) / 255
	extend_vision_frame = cv2.resize(extend_vision_frame, size, interpolation = cv2.INTER_AREA)
	color_difference = extend_vision_frame_raw - extend_vision_frame
	return color_difference


def normalize_color_difference(color_difference : VisionFrame, color_difference_mask : Mask, extend_vision_frame : VisionFrame) -> VisionFrame:
	color_difference = cv2.resize(color_difference, extend_vision_frame.shape[:2][::-1], interpolation = cv2.INTER_CUBIC)
	color_difference_mask = 1 - color_difference_mask.clip(0, 0.75)
	extend_vision_frame = extend_vision_frame.astype(np.float32) / 255
	extend_vision_frame += color_difference * color_difference_mask
	extend_vision_frame = extend_vision_frame.clip(0, 1)
	extend_vision_frame = np.multiply(extend_vision_frame, 255).astype(np.uint8)
	return extend_vision_frame


def prepare_direction(direction : int) -> NDArray[Any]:
	direction = np.interp(float(direction), [ -100, 100 ], [ 2.5, -2.5 ]) #type:ignore[assignment]
	return np.array(direction).astype(np.float32)


def prepare_vision_frame(vision_frame: VisionFrame) -> VisionFrame:
    # Reverse the color channels from BGR to RGB and normalize pixel values to the range [0, 1]
    vision_frame = vision_frame[:, :, ::-1] / 255.0
    
    # Normalize the pixel values to have zero mean and unit variance (center around 0 with range [-1, 1])
    vision_frame = (vision_frame - 0.5) / 0.5
    
    # Rearrange the axes from [height, width, channels] to [channels, height, width]
    # and add an extra dimension to create a batch dimension (shape: [1, channels, height, width])
    vision_frame = np.expand_dims(vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
    
    # Return the prepared vision frame
    return vision_frame


def normalize_extend_frame(extend_vision_frame : VisionFrame) -> VisionFrame:
	extend_vision_frame = np.clip(extend_vision_frame, -1, 1)
	extend_vision_frame = (extend_vision_frame + 1) / 2
	extend_vision_frame = extend_vision_frame.transpose(1, 2, 0).clip(0, 255)
	extend_vision_frame = (extend_vision_frame * 255.0)
	extend_vision_frame = extend_vision_frame.astype(np.uint8)[:, :, ::-1]
	extend_vision_frame = cv2.pyrDown(extend_vision_frame)
	return extend_vision_frame


def get_reference_frame(source_face : Face, target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	return modify_age(target_face, temp_vision_frame)


def process_frame(inputs : AgeModifierInputs) -> VisionFrame:
	reference_faces = inputs.get('reference_faces')
	target_vision_frame = inputs.get('target_vision_frame')

	many_faces = sort_and_filter_faces(get_many_faces([ target_vision_frame ]))

	if state_manager.get_item('face_selector_mode') == 'many':
		if many_faces:
			for target_face in many_faces:
				target_vision_frame = modify_age(target_face, target_vision_frame)
	if state_manager.get_item('face_selector_mode') == 'one':
		target_face = get_one_face(many_faces)
		if target_face:
			target_vision_frame = modify_age(target_face, target_vision_frame)
	if state_manager.get_item('face_selector_mode') == 'reference':
		similar_faces = find_similar_faces(many_faces, reference_faces, state_manager.get_item('reference_face_distance'))
		if similar_faces:
			for similar_face in similar_faces:
				target_vision_frame = modify_age(similar_face, target_vision_frame)
	return target_vision_frame


def process_frames(source_path : List[str], queue_payloads : List[QueuePayload], update_progress : UpdateProgress) -> None:
	reference_faces = get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else None

	for queue_payload in process_manager.manage(queue_payloads):
		target_vision_path = queue_payload['frame_path']
		target_vision_frame = read_image(target_vision_path)
		output_vision_frame = process_frame(
		{
			'reference_faces': reference_faces,
			'target_vision_frame': target_vision_frame
		})
		write_image(target_vision_path, output_vision_frame)
		update_progress(1)


def process_image(source_path : str, target_path : str, output_path : str) -> None:
	reference_faces = get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else None
	target_vision_frame = read_static_image(target_path)
	output_vision_frame = process_frame(
	{
		'reference_faces': reference_faces,
		'target_vision_frame': target_vision_frame
	})
	write_image(output_path, output_vision_frame)


def process_video(source_paths : List[str], temp_frame_paths : List[str]) -> None:
	processors.multi_process_frames(None, temp_frame_paths, process_frames)
