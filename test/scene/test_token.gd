extends Node2D

func _ready():
	$play_animation.connect("button_down", _on_play_animation)
	
func _on_play_animation():
	$token.animate.play("melee")
