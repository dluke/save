extends Label

@onready var animate = $animate

@export var font_color := Color(236, 72, 73)

func _ready():
	animate.play("RESET")
	set("theme_override_colors/font_color", font_color)

func show_damage(value):
	set_text(str(value))
	animate.play("float")
