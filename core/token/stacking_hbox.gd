extends Control

# implement something like hbox container except children are stacked 
var overlap_distance = 4

var symbol = TextureRect.new()

func _ready():
	symbol.set_texture(load("res://assets/placeholder/icon/stats/walking-boot.png"))
	symbol.set_expand_mode(TextureRect.EXPAND_IGNORE_SIZE)
	symbol.set_size(Vector2(12,12))
	restack()
	
func set_value(x):
	for child in get_children():
		child.queue_free()
	for _i in range(x):
		add_child(symbol.duplicate())
	restack()
	
func restack():
	# children drawn from left to right, stacked form right to left
	var children = get_children()
	children.reverse()
	for i in range(children.size()):
		var child = children[i]
		var symbol_size = child.get_size()
		var offset = i * overlap_distance
		child.set_position(Vector2(-symbol_size.x - offset, 0))
