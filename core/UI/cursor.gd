extends Node

# TODO should be an autoload?

## sprites
var texture = [
	preload("res://assets/placeholder/icon/cursor/cursor_none.png"),
	preload("res://assets/placeholder/icon/cursor/zoom_alt.png"),
	preload("res://assets/placeholder/icon/cursor/hand_open.png"),
	preload("res://assets/placeholder/icon/cursor/hand_closed.png"),
	preload("res://assets/placeholder/icon/cursor/tool_sword_b.png"),
	preload("res://assets/placeholder/icon/cursor/tool_bow.png"),
	preload("res://assets/placeholder/icon/cursor/tool_wand.png")
	]

# cursor type
enum {ARROW, INSPECT, HANDOPEN, HANDCLOSED, TARGETMELEE, TARGETRANGED, WAND}

@export var m_cursor_scale = Vector2(0.5, 0.5)

@onready var m_sprite = $sprite

func _ready():
	Input.set_mouse_mode(Input.MOUSE_MODE_HIDDEN)
	set_texture(ARROW)
	m_sprite.set_visible(true)
	m_sprite.set_scale(m_cursor_scale)

func _process(delta):
	var mpos = get_viewport().get_mouse_position()
	m_sprite.set_position(mpos)
	
func set_texture(cursor_type):
	if m_sprite:
		m_sprite.set_texture(texture[cursor_type])
	
