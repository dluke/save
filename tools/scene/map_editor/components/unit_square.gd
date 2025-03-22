extends PanelContainer

# map editor component

signal pickup_unit

@onready var texture_rect = $TextureRect

var data

func set_data(x):
	data = x
	var sprite_texture = load(data['sprite_path'])
	texture_rect.texture = sprite_texture

func _gui_input(event):

	if event is InputEventMouseButton and event.pressed:
		pickup_unit.emit(data)
		
