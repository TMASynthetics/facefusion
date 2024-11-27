import os
import tempfile
from typing import Optional

import gradio

from facefusion import state_manager, wording
from facefusion.uis.core import register_ui_component

OUTPUT_PATH_TEXTBOX : Optional[gradio.Textbox] = None
OUTPUT_IMAGE : Optional[gradio.Image] = None
OUTPUT_VIDEO : Optional[gradio.Video] = None


def render() -> None:
	global OUTPUT_PATH_TEXTBOX
	global OUTPUT_FILE
	global OUTPUT_IMAGE
	global OUTPUT_VIDEO

	if not state_manager.get_item('output_path'):
		state_manager.set_item('output_path', tempfile.gettempdir())
	OUTPUT_PATH_TEXTBOX = gradio.Textbox(
		label = wording.get('uis.output_path_textbox'),
		value = state_manager.get_item('output_path'),
		max_lines = 1
	)
	OUTPUT_FILE = gradio.File(
		label = wording.get('uis.output_file'),
		interactive=False,
		file_count = 'single',
		file_types = ['image','video'],
		visible = False,  # Visible only for videos
	)
	OUTPUT_IMAGE = gradio.Image(
		label = wording.get('uis.output_image_or_video'),
		visible = False
	)
	OUTPUT_VIDEO = gradio.Video(
		interactive=False,
		label = wording.get('uis.output_image_or_video')
	)


def listen() -> None:
	OUTPUT_PATH_TEXTBOX.change(update_output_path, inputs = OUTPUT_PATH_TEXTBOX)
	register_ui_component('output_file', OUTPUT_FILE)
	register_ui_component('output_image', OUTPUT_IMAGE)
	register_ui_component('output_video', OUTPUT_VIDEO)	
	OUTPUT_VIDEO.change( update_video_file, inputs=OUTPUT_VIDEO, outputs=[OUTPUT_FILE, OUTPUT_FILE] )

	
def update_video_file(video_metadata):
	# video_metadata is a dictionary with a "name" key that holds the file path
	if video_metadata:
		mov_file_name = os.path.splitext(os.path.basename(video_metadata))[0]
		mov_base_path = state_manager.get_item('output_path')
		mov_full_path =  os.path.join(mov_base_path, mov_file_name + ".mov")
		if os.path.isfile(mov_full_path):	
			return mov_full_path, gradio.update(visible=True)
		else:
			return video_metadata, gradio.update(visible=True)
	else:
		return None, gradio.update(visible=False)


def update_output_path(output_path : str) -> None:
	state_manager.set_item('output_path', output_path)
