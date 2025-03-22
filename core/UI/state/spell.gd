extends State

# Notes
# Use a spell

## setup

@onready var Cursor = %Cursor
@onready var Message = %Message

var spell : SpellResource

## enter/exit

func enter():
	print('enter spell state')
	Cursor.set_texture(Cursor.WAND)

func exit():
	spell = null

## handle input

func input(event):
	
	if event is InputEventMouseMotion:
		var hexpt = node.position_to_hexpt(event.position)
		var tile = node.tiles.get(hexpt)
		if tile and tile.slot:
			if spell.can_target(tile.slot):
				Message.push('%s can target %s' % [str(spell), str(tile.slot)])

	if event is InputEventMouseButton:
		var hexpt = node.position_to_hexpt(event.position)
		var tile = node.tiles.get(hexpt)
		if event.button_index == MOUSE_BUTTON_LEFT and !event.is_pressed():
			var success = false
			if tile and tile.slot:
				var token = tile.slot
				if spell.can_target(token):
					spell.set_target(token)
					spell.forward()
					spell.play()
			if !success:
				fsm.back()
					
		if event.button_index == MOUSE_BUTTON_RIGHT and event.is_pressed():
			fsm.back()
	
