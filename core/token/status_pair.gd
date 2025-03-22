extends HBoxContainer

@onready var label =  $RichTextLabel
@onready var texture_rect = $CenterContainer/TextureRect

@export var text: String = ""

func set_text(value):
	text = value
	if label:
		label.set_text(text)

func _ready():
	label.set_text(text)
