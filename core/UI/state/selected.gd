extends SharedState
class_name Selected

# Notes

## setup


## enter/exit

func enter():
	token = node.m_selected_item
	Cursor.set_texture(Cursor.ARROW)
	Message.push('Selected %s' % str(token))
	_shared_enter()
	
	# debugging
	# find a target
#	var action_list = token.evaluate()
#	var viable = token.pathing.viable_far(token)
#	if viable.is_empty():
#		return [] # do nothing
#	var far_tile = token.pathing.min_key(viable)
#	var far_path = token.pathing.path_to(far_tile)
#	var path_indicator = node.draw_path(far_path)
	

func exit():
	Message.clear()
	node.clear_movement_range()
	node.clear_aim_range()
	node.clear_indicator(selected_coord)
	node.clear_glyphs()
	node.m_selected_item = null

## handle input

func input(event):
#	if event is InputEventKey and event.pressed:
#		if event.keycode == KEY_M:
#			fsm.to_state(fsm.inspect)


	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_E:
			fsm.to_state(fsm.base)
			node.end_turn.emit()
		if event.keycode == KEY_ESCAPE:
			fsm.to_state(fsm.base)

	if event is InputEventMouseButton:
		var hexpt = node.pixel_to_cube(node.to_local(event.position)).get_axial()
		var tile = node.tiles.get(hexpt)
		var other = node.tiles[hexpt].slot if tile else null
		# left click
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed == true:
			if other and token.capable_ranged_attack() and token.can_shoot(other):
				token.animate_ranged_attack(other) # form the animation first in case target is killed and removed
				token.ranged_attack(other)
				fsm.to_state(fsm.base)
			elif other and token.can_attack(other): # move and attack
				# move 
				var path = get_directional_path(other, node.to_local(event.position))
				if path:
					if path[0] != token.tile:
						node.move_path(token, path, token.pathing.dist[path[0]])

					# attack
					node.snap(token, token.tile)
					token.attack_or_counter(other)
				fsm.to_state(fsm.base)
			elif tile and tile != token.tile and tile in node.m_current_movement:
				# move
				var path = token.pathing.path_to(tile) # path is list of tiles
				node.move_path(token, path, token.pathing.dist[tile])
				fsm.exit()
				node.m_selected_item = token
				fsm.enter_state(fsm.selected)
			elif tile and tile == token.tile:
				# clicking same selected tile goes to drag state
				node.m_drag_item = token
				fsm.to_state(fsm.grabbed) 
			elif tile and tile.slot and tile.slot.get_team() == token.get_team():
				node.m_drag_item = tile.slot
				fsm.to_state(fsm.grabbed)
			else:
				fsm.to_state(fsm.base)
		# right click
		if event.button_index == MOUSE_BUTTON_RIGHT and event.pressed == true:
			if tile and tile.slot == node.m_selected_item:
				# deselect
				fsm.exit()
				node.m_selected_item = tile.slot
				fsm.enter_state(fsm.base) 
			elif tile and tile.slot and tile.slot.get_team() == token.get_team():
				# select friendly
				fsm.exit()
				node.m_selected_item = tile.slot
				fsm.enter_state(fsm.selected) 
			elif tile and tile.slot and tile.slot.get_team() != token.get_team():
				# select enemy
				fsm.exit()
				node.m_selected_item = tile.slot
				fsm.enter_state(fsm.selected) 
			else:
				fsm.to_state(fsm.base)


	# Note. combine with grabbed.gd
	if event is InputEventMouseMotion:
		var hexpt = node.pixel_to_cube(node.to_local(event.position)).get_axial()
		var other = node.tiles[hexpt].slot if node.tiles.get(hexpt) else null
		if other and token.capable_ranged_attack() and token.can_shoot(other):
			Cursor.set_texture(Cursor.TARGETRANGED)
		elif other and token.can_attack(other):
			var path = get_directional_path(other, node.to_local(event.position))
			if path:
				Cursor.set_texture(Cursor.TARGETMELEE)
				# directional move/attack

				node.clear_glyphs()
				if path and path[0] != token.tile:
					node.add_glyph(path[0], node.Glyph.DOWNARROW)
		else:
			Cursor.set_texture(Cursor.ARROW)
			node.clear_glyphs()
