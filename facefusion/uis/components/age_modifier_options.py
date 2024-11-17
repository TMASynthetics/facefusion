from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, wording
from facefusion.common_helper import calc_float_step
from facefusion.processors import choices as processors_choices
from facefusion.processors.core import load_processor_module
from facefusion.processors.typing import AgeModifierModel
from facefusion.uis.core import get_ui_component, register_ui_component

AGE_MODIFIER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
AGE_MODIFIER_DIRECTION_SLIDER : Optional[gradio.Slider] = None
AGE_MODIFIER_SOURCE_AGE_SLIDER : Optional[gradio.Slider] = None
AGE_MODIFIER_TARGET_AGE_SLIDER : Optional[gradio.Slider] = None
AGE_MODIFIER_STRIDE_SLIDER : Optional[gradio.Slider] = None
AGE_MODIFIER_MASK_TYPE : Optional[gradio.Checkbox] = None

def render() -> None:
	global AGE_MODIFIER_MODEL_DROPDOWN
	global AGE_MODIFIER_DIRECTION_SLIDER
	global AGE_MODIFIER_SOURCE_AGE_SLIDER
	global AGE_MODIFIER_TARGET_AGE_SLIDER
	global AGE_MODIFIER_STRIDE_SLIDER
	global AGE_MODIFIER_MASK_TYPE

	AGE_MODIFIER_MODEL_DROPDOWN = gradio.Dropdown(
		label = wording.get('uis.age_modifier_model_dropdown'),
		choices = processors_choices.age_modifier_models,
		value = state_manager.get_item('age_modifier_model'),
		visible = 'age_modifier' in state_manager.get_item('processors')
	)
	AGE_MODIFIER_DIRECTION_SLIDER = gradio.Slider(
		label = wording.get('uis.age_modifier_direction_slider'),
		value = state_manager.get_item('age_modifier_direction'),
		step = calc_float_step(processors_choices.age_modifier_direction_range),
		minimum = processors_choices.age_modifier_direction_range[0],
		maximum = processors_choices.age_modifier_direction_range[-1],
		visible = 'age_modifier' in state_manager.get_item('processors') and 'fran' not in state_manager.get_item('age_modifier_model')
	)
	AGE_MODIFIER_SOURCE_AGE_SLIDER = gradio.Slider(
		label = wording.get('uis.age_modifier_source_age_slider'),
		value = state_manager.get_item('age_modifier_source_age'),
		step = calc_float_step(processors_choices.age_modifier_source_age_range),
		minimum = processors_choices.age_modifier_source_age_range[0],
		maximum = processors_choices.age_modifier_source_age_range[-1],
		visible = 'age_modifier' in state_manager.get_item('processors') and 'fran' in state_manager.get_item('age_modifier_model')
	)
	AGE_MODIFIER_TARGET_AGE_SLIDER = gradio.Slider(
		label = wording.get('uis.age_modifier_target_age_slider'),
		value = state_manager.get_item('age_modifier_target_age'),
		step = calc_float_step(processors_choices.age_modifier_target_age_range),
		minimum = processors_choices.age_modifier_target_age_range[0],
		maximum = processors_choices.age_modifier_target_age_range[-1],
		visible = 'age_modifier' in state_manager.get_item('processors') and 'fran' in state_manager.get_item('age_modifier_model')
	)
	AGE_MODIFIER_STRIDE_SLIDER = gradio.Slider(
		label = wording.get('uis.age_modifier_stride_slider'),
		value = state_manager.get_item('age_modifier_stride'),
		step = calc_float_step(processors_choices.age_modifier_stride_range),
		minimum = processors_choices.age_modifier_stride_range[0],
		maximum = processors_choices.age_modifier_stride_range[-1],
		visible = 'age_modifier' in state_manager.get_item('processors') and 'fran' in state_manager.get_item('age_modifier_model')
	)
	AGE_MODIFIER_MASK_TYPE = gradio.CheckboxGroup(
		label = wording.get('uis.age_modifier_mask_type'),
		choices = ['FRAN', 'CUSTOM'],
		value = 'FRAN', 
		visible = 'age_modifier' in state_manager.get_item('processors') and 'fran' in state_manager.get_item('age_modifier_model')
	)

	register_ui_component('age_modifier_model_dropdown', AGE_MODIFIER_MODEL_DROPDOWN)
	register_ui_component('age_modifier_direction_slider', AGE_MODIFIER_DIRECTION_SLIDER)
	register_ui_component('age_modifier_source_age_slider', AGE_MODIFIER_SOURCE_AGE_SLIDER)
	register_ui_component('age_modifier_target_age_slider', AGE_MODIFIER_TARGET_AGE_SLIDER)
	register_ui_component('age_modifier_stride_slider', AGE_MODIFIER_STRIDE_SLIDER)
	register_ui_component('age_modifier_mask_type', AGE_MODIFIER_MASK_TYPE)

def listen() -> None:
	AGE_MODIFIER_MODEL_DROPDOWN.change(update_age_modifier_model, inputs = AGE_MODIFIER_MODEL_DROPDOWN, outputs = [ AGE_MODIFIER_MODEL_DROPDOWN, AGE_MODIFIER_DIRECTION_SLIDER, AGE_MODIFIER_SOURCE_AGE_SLIDER, AGE_MODIFIER_TARGET_AGE_SLIDER, AGE_MODIFIER_STRIDE_SLIDER, AGE_MODIFIER_MASK_TYPE ])
	AGE_MODIFIER_DIRECTION_SLIDER.release(update_age_modifier_direction, inputs = AGE_MODIFIER_DIRECTION_SLIDER)
	AGE_MODIFIER_SOURCE_AGE_SLIDER.release(update_age_modifier_source_age, inputs = AGE_MODIFIER_SOURCE_AGE_SLIDER)
	AGE_MODIFIER_TARGET_AGE_SLIDER.release(update_age_modifier_target_age, inputs = AGE_MODIFIER_TARGET_AGE_SLIDER)
	AGE_MODIFIER_STRIDE_SLIDER.release(update_age_modifier_stride, inputs = AGE_MODIFIER_STRIDE_SLIDER)
	AGE_MODIFIER_MASK_TYPE.change(update_age_modifier_mask_type, inputs = AGE_MODIFIER_MASK_TYPE, outputs = AGE_MODIFIER_MASK_TYPE)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ AGE_MODIFIER_MODEL_DROPDOWN, AGE_MODIFIER_DIRECTION_SLIDER, AGE_MODIFIER_SOURCE_AGE_SLIDER, AGE_MODIFIER_TARGET_AGE_SLIDER, AGE_MODIFIER_STRIDE_SLIDER, AGE_MODIFIER_MASK_TYPE ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.CheckboxGroup]:
	has_age_modifier = 'age_modifier' in processors
	has_fran = 'fran' in state_manager.get_item('age_modifier_model')
	print(has_age_modifier, has_fran)
	return gradio.Dropdown(visible = has_age_modifier), gradio.Slider(visible = has_age_modifier and not has_fran), gradio.Slider(visible = has_age_modifier and has_fran), gradio.Slider(visible = has_age_modifier and has_fran), gradio.Slider(visible= has_age_modifier and has_fran), gradio.CheckboxGroup(visible = has_age_modifier and has_fran)


def update_age_modifier_model(age_modifier_model : AgeModifierModel) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.CheckboxGroup]:
	age_modifier_module = load_processor_module('age_modifier')
	age_modifier_module.clear_inference_pool()
	state_manager.set_item('age_modifier_model', age_modifier_model)

	has_age_modifier = 'age_modifier' in state_manager.get_item('processors')
	has_fran = 'fran' in state_manager.get_item('age_modifier_model')
	
	if age_modifier_module.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('age_modifier_model')), gradio.Slider(visible = has_age_modifier and not has_fran), gradio.Slider(visible = has_age_modifier and has_fran), gradio.Slider(visible = has_age_modifier and has_fran), gradio.Slider(visible= has_age_modifier and has_fran), gradio.CheckboxGroup(visible = has_age_modifier and has_fran)
	return gradio.Dropdown(visible = has_age_modifier), gradio.Slider(visible = has_age_modifier and not has_fran), gradio.Slider(visible = has_age_modifier and has_fran), gradio.Slider(visible = has_age_modifier and has_fran), gradio.Slider(visible= has_age_modifier and has_fran), gradio.CheckboxGroup(visible = has_age_modifier and has_fran)


def update_age_modifier_direction(age_modifier_direction : float) -> None:
	state_manager.set_item('age_modifier_direction', int(age_modifier_direction))


def update_age_modifier_source_age(age_modifier_source_age : float) -> None:
	state_manager.set_item('age_modifier_source_age', int(age_modifier_source_age))


def update_age_modifier_target_age(age_modifier_target_age : float) -> None:
	state_manager.set_item('age_modifier_target_age', int(age_modifier_target_age))


def update_age_modifier_stride(age_modifier_stride : int) -> None:
	state_manager.set_item('age_modifier_stride', int(age_modifier_stride))
	

def update_age_modifier_mask_type(age_modifier_mask : str) -> gradio.CheckboxGroup:
	state_manager.set_item('age_modifier_mask_type', age_modifier_mask)
	return gradio.CheckboxGroup(value = state_manager.get_item('age_modifier_mask_type'))