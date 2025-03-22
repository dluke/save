extends State

# Notes

## setup

@onready var Cursor = %Cursor
@onready var Message = %Message
@onready var StatPanel = %StatPanel

## enter/exit

func enter():
	Cursor.set_texture(Cursor.INSPECT)
	Message.push("Inspect State.")
	
func exit():
	StatPanel.set_visible(false)
	Message.clear()

## handle input

func input(event):
	
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_E and event.pressed:
			fsm.back()
			node.end_turn.emit()
			
	# exit inspect state by toggling
	if event is InputEventKey and event.pressed:
		if event.keycode in [KEY_M, KEY_ESCAPE]:
			fsm.back()
			
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
			var hexpt = node.position_to_hexpt(event.position)
			var tiles = node.tiles
			if hexpt in tiles and tiles[hexpt].slot:
				var token = tiles[hexpt].slot
				node.m_selected_item = token
				fsm.to_state(fsm.selected)


	if event is InputEventMouseMotion:
		var hexpt = node.position_to_hexpt(event.position)
		var tiles = node.tiles
		if hexpt in tiles and tiles[hexpt].slot:
			var token = tiles[hexpt].slot
			Message.push('%s' % str(token))
			_show_stat_panel(token)
		else:
			Message.clear()
			_hide_stat_panel()

func _show_stat_panel(token):
	StatPanel.set_visible(true)
	# move to position
	StatPanel.set_position(token.get_global_position())
	# update
	StatPanel.update(token)


func _hide_stat_panel():
	StatPanel.set_visible(false)
	
