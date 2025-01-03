import gradio

from facefusion import state_manager
from facefusion.uis.components import about, age_modifier_options, common_options, execution, execution_queue_count, execution_thread_count, expression_restorer_options, face_debugger_options, face_detector, face_editor_options, face_enhancer_options, face_landmarker, face_masker, face_selector, face_swapper_options, frame_colorizer_options, frame_enhancer_options, instant_runner, job_manager, job_runner, lip_syncer_options, memory, output, output_options, preview, processors, source, target, temp_frame, terminal, trim_frame, ui_workflow


def pre_check() -> bool:
	state_manager.set_item('advanced_user', False)
	#age_modifier_options.render() # avoid Key Error at launch
	return True


def toggle_source_visibility(selected_processors):
    # check if one processor need the source window
    requires_source = any(
        processor in processors.processors_requiring_source() for processor in selected_processors
    )
    return gradio.update(visible=requires_source)

def render() -> gradio.Blocks:
	with gradio.Blocks() as layout:
		# Advanced_user visibility functionnalies and blocks
		def toggle_advanced_user(current_state):
			state_manager.set_item('advanced_user', current_state)
			advanced_user = state_manager.get_item('advanced_user') == True
			return [
				gradio.update(visible=True), # basic_block1
				gradio.update(visible=advanced_user), # advanced_block1
				gradio.update(visible=True), # basic_block2
				gradio.update(visible=advanced_user), # advanced_block2
				gradio.update(visible=advanced_user), # advanced_block3
				gradio.update(visible=advanced_user and 'age_modifier' in state_manager.get_item('processors') and 'fran' in state_manager.get_item('age_modifier_model')), # age_modifier_stride slider
				gradio.update(visible=advanced_user and 'face_swapper' in state_manager.get_item('processors')), # face_swapper_model_dropdown
				gradio.update(visible=advanced_user), # log_level dropdown
				gradio.update(visible=advanced_user), # face_mask_blur slider
				gradio.update(visible=advanced_user), # output output_path
			]
		
		with gradio.Row():
			# Column 1
			with gradio.Column(scale=4):
				with gradio.Blocks():
					about.render()
				
				# Switch for Advanced User
				with gradio.Blocks():
					advanced_user_switch = gradio.Checkbox(
					label="Advanced User",
					value=state_manager.get_item('advanced_user') == True,
					interactive=True
					)

				with gradio.Blocks():
					processors.render()

				with gradio.Group(visible=True) as basic_block1:
					with gradio.Blocks():
						age_modifier_options.render()
					with gradio.Blocks():
						expression_restorer_options.render()
					with gradio.Blocks():
						face_debugger_options.render()
					with gradio.Blocks():
						face_editor_options.render()
					with gradio.Blocks():
						face_enhancer_options.render()
					with gradio.Blocks():
						face_swapper_options.render()
					with gradio.Blocks():
						frame_colorizer_options.render()
					with gradio.Blocks():
						frame_enhancer_options.render()
					with gradio.Blocks():
						lip_syncer_options.render()

				with gradio.Group(visible=state_manager.get_item('advanced_user')) as advanced_block1:
					with gradio.Blocks():
						execution.render()
						execution_thread_count.render()
						execution_queue_count.render()
					with gradio.Blocks():
						memory.render()

				with gradio.Group(visible=True) as basic_block2:
					with gradio.Blocks():
						temp_frame.render()
					with gradio.Blocks():
						output_options.render()

			# Column 2
			with gradio.Column(scale = 4):
				with gradio.Group(visible=True) as source_group:
					source.render()
				with gradio.Blocks():
					target.render()
				with gradio.Blocks():
					output.render()
				with gradio.Blocks():
					terminal.render()

				with gradio.Group(visible=state_manager.get_item('advanced_user')) as advanced_block2:
					with gradio.Blocks():
						ui_workflow.render()
				
				with gradio.Blocks():
					instant_runner.render()
					job_runner.render()
					job_manager.render()

			# Column 3
			with gradio.Column(scale = 7):
				with gradio.Blocks():
					preview.render()
				with gradio.Blocks():
					trim_frame.render()
				with gradio.Blocks():
					face_selector.render()
				with gradio.Blocks():
					face_masker.render()

				with gradio.Group(visible=state_manager.get_item('advanced_user')) as advanced_block3:
					with gradio.Blocks():
						face_detector.render()
					with gradio.Blocks():
						face_landmarker.render()
					with gradio.Blocks():
						common_options.render()

		# Groups update
		advanced_user_switch.change(
			toggle_advanced_user,
			inputs=[advanced_user_switch],
			outputs=[basic_block1, advanced_block1, basic_block2, advanced_block2, advanced_block3, 
			age_modifier_options.AGE_MODIFIER_STRIDE_SLIDER,
			face_swapper_options.FACE_SWAPPER_MODEL_DROPDOWN,
			terminal.LOG_LEVEL_DROPDOWN,
			face_masker.FACE_MASK_BLUR_SLIDER,
			output.OUTPUT_PATH_TEXTBOX
			]
		)

		# Connect Processors to Source
		processors.PROCESSORS_CHECKBOX_GROUP.change(
			toggle_source_visibility,
			inputs=[processors.PROCESSORS_CHECKBOX_GROUP],
			outputs=[source_group]
		)
	return layout


def listen() -> None:
	processors.listen()
	age_modifier_options.listen()
	expression_restorer_options.listen()
	face_debugger_options.listen()
	face_editor_options.listen()
	face_enhancer_options.listen()
	face_swapper_options.listen()
	frame_colorizer_options.listen()
	frame_enhancer_options.listen()
	lip_syncer_options.listen()
	temp_frame.listen()
	output_options.listen()
	source.listen()
	target.listen()
	output.listen()
	job_runner.listen()
	job_manager.listen()
	terminal.listen()
	preview.listen()
	trim_frame.listen()
	face_selector.listen()
	face_masker.listen()
	execution.listen()
	execution_thread_count.listen() 
	execution_queue_count.listen()
	memory.listen()
	instant_runner.listen()
	face_detector.listen()
	face_landmarker.listen()
	common_options.listen()



def run(ui : gradio.Blocks) -> None:
	# ui.launch(favicon_path = 'facefusion.ico', inbrowser = state_manager.get_item('open_browser'))
	ui.launch(inbrowser = state_manager.get_item('open_browser'))