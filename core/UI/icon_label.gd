extends HBoxContainer

@onready var label = $label
@onready var icon = $icon

@export var texture: Texture2D
@export var value: int

func _ready():
	icon.set_texture(texture)
	label.set_text(str(value))
	
func set_value(x):
	label.set_text(str(x))
