extends SharedState

# Notes

## enter/exit


func enter():
	token = node.m_drag_item
	Message.push('Grabbed %s' % str(token))
	token.set_z_index(1)
	_shared_enter()


func exit():
	Message.clear()
	node.clear_movement_range()
	node.clear_aim_range()
	node.clear_indicator(selected_coord)
	token.set_z_index(0)
	node.clear_glyphs()
	if token.tile:
		node.snap(token, token.tile)
	node.m_drag_item = null

## handle input

func input(event):
	
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_E and event.pressed:
			fsm.to_state(fsm.base)
			node.end_turn.emit()

	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed == false:
			var hexpt = node.pixel_to_cube(node.to_local(event.position)).get_axial()
			var tile = node.tiles.get(hexpt)
			var other = node.tiles[hexpt].slot if tile else null
			if other and token.can_attack(other): # move and attack
				# move 
				var path = get_directional_path(other, node.to_local(event.position))
				if path:
					if path[0] != token.tile:
						node.move_path(token, path, token.pathing.dist[path[0]])

					# attack
					node.snap(token, token.tile)
					token.attack_or_counter(other)
				fsm.to_state(fsm.base)
			elif tile and token.can_move(tile) and tile in node.m_current_movement:
				# move
				var path = token.pathing.path_to(tile) # path is list of tiles
				node.move_path(token, path, token.pathing.dist[tile])
				fsm.exit()
				node.m_selected_item = token
				fsm.enter_state(fsm.selected)
			elif tile and tile == token.tile:
				# drop back on the same tile enters select state
				node.m_selected_item = token
				node.drop_at(token.tile)
				fsm.to_state(fsm.selected)
			else:
				# return
				node.drop_at(token.tile)
				fsm.to_state(fsm.base)

	if event is InputEventMouseMotion:
		token.set_position(node.to_local(event.position))
		var hexpt = node.pixel_to_cube(node.to_local(event.position)).get_axial()
		var other = node.tiles[hexpt].slot if node.tiles.get(hexpt) else null
		if other and token.can_attack(other):
			Cursor.set_texture(Cursor.TARGETMELEE)
			# directional move/attack
			var path = get_directional_path(other, node.to_local(event.position))
			node.clear_glyphs()
			if path and path[0] != token.tile:
				node.add_glyph(path[0], node.Glyph.DOWNARROW)
		else:
			Cursor.set_texture(Cursor.HANDCLOSED)
			node.clear_glyphs()

func get_directional_path(other, local_position):
	# find the path that moves the hex closest to the cursor
	var path = null
	# calculate distances to adjacent tiles
	var min_dist = INF
	var close_tile = null
	for tile in other.tile.adjacent:
		if !m_viable_tiles.has(tile):
			continue
		var d = local_position.distance_to(node.cube_to_pixel(tile.cube))
		if d < min_dist:
			close_tile = tile
			min_dist = d
	if close_tile:
		path = token.pathing.path_to(close_tile)
	return path


