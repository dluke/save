extends Control

## setup

@export var value := 5.0
@export var max_value := 10.0
@export var inter_value = 1.0:
	set(_v):
		inter_value = _v
		queue_redraw()
@export var color := Color(1,0,0)

var prev_value
var border_color := Color(0,0,0)
var inter_color = color.lightened(0.5)

@onready var animate = $animate

func set_max_value(_v):
	max_value = _v
	
## setup
	
func init(_value, _max_value):
	max_value = _max_value
	value = _value
	prev_value = value
	queue_redraw()

## update

func fill():
	value = max_value
	queue_redraw()

func update(_value):
	prev_value = value
	if _value < value:
		animate.play("value_down")
	value = _value
	queue_redraw()

## rendering

func _draw():
	var _p = Vector2(0,0)
	var _size = get_size()
	# border
	draw_rect(Rect2(_p, _size), border_color, true)
	
	# inter bar
	_p = _p + Vector2(1, 1)
	_size = _size - Vector2(2, 2)
	var new_x = value/max_value * _size.x
	if prev_value != null:
		var prev_x = prev_value/max_value * _size.x
		var inter_x = new_x + inter_value * (prev_x - new_x)
		_size.x = inter_x
		draw_rect(Rect2(_p, _size), inter_color, true)

	# bar
	_size.x = new_x
	draw_rect(Rect2(_p, _size), color, true)

	
