extends State

# Notes
# No units are selected. 
# Clicking on a unit without dragging selects it.
# Clicking and dragging enters the select state with a unit grabbed.

## setup

@onready var Cursor = %Cursor
@onready var Message = %Message

## enter/exit

func enter():
	Cursor.set_texture(Cursor.ARROW)

func exit():
	pass

## handle input

func input(event):
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_E and event.pressed:
			node.end_turn.emit()
			
		if event.keycode == KEY_M:
			fsm.to_state(fsm.inspect)
		

	if event is InputEventMouseButton:
		var hexpt = node.position_to_hexpt(event.position)
		var tile = node.tiles.get(hexpt)
		if event.button_index == MOUSE_BUTTON_LEFT and event.is_pressed():
			if tile and tile.slot:
				var token = tile.slot
				if node.Model.active_team == token.get_team() and token.get_team().player_control:
					node.pickup(tile.slot)
					fsm.to_state(fsm.grabbed)
				else:
					node.m_selected_item = token
					fsm.to_state(fsm.selected)
					
		if event.button_index == MOUSE_BUTTON_RIGHT and event.is_pressed():
			if tile and tile.slot:
				var token = tile.slot
				node.m_selected_item = token
				fsm.to_state(fsm.selected)
