extends Node2D

## setup
@onready var hexmap = $grid_layer/TileMap
@onready var Message = %Message
@onready var terrain_option = $OptionButton
@onready var background = $render_background

@onready var unit_panel = $UILayer/UnitPanel



var tile_size
var m_hexes : Dictionary
var terrain_type = Dictionary()
var rng : RandomNumberGenerator

# poor mans state machine is just a variable and a match statement
enum {BASE, DRAG_UNIT, PLACE_TERRAIN}
var state = BASE 
var drag_sprite = null
var drag_data = null

func _ready():
	tile_size = hexmap.hex_size
	m_hexes = grid_hexes()
	$SaveTerrain.pressed.connect(save_terrain)
	$UILayer/SaveUnitLayout.pressed.connect(save_layout)
	rng = background.rng # use the random number generator of the background layer
	connect_unit_panel(unit_panel)
	terrain_option.item_selected.connect(_on_terrain_selected)

func _on_terrain_selected(idx : int):
	if idx == 0:
		state = BASE
	else:
		state = PLACE_TERRAIN


func grid_hexes():
	var hexes = Dictionary()
	var left = hexmap.Cube.new(0, 0, 0)
	var east = hexmap.hex_direction_vector[0]
	var down = [hexmap.hex_direction_vector[1], hexmap.hex_direction_vector[2]]
	for i in range(8):
		var current = left
		for j in range(8):
			hexes[current.get_axial()] = null
			current = current.add(east)
		left = left.add(down[i % 2])
	return hexes
		
func grid_coord(hexpt):
	return hexmap.local_to_map(hexmap.point_transform * hexpt)
		
		
## input

func _input(event):

	if event is InputEventMouseMotion:
		var cube = hexmap.pixel_to_cube(hexmap.to_local(event.position))
		var hexpt = cube.get_axial()
		if m_hexes.has(hexpt):
			Message.push(str(cube.get_axial()))
		
		match state:
			BASE:
				pass
			DRAG_UNIT:
				drag_sprite.set_position(event.position)

	if event is InputEventMouseButton:
		var cube = hexmap.pixel_to_cube(hexmap.to_local(event.position))
		var hexpt = cube.get_axial()
			
		if event.button_index == MOUSE_BUTTON_RIGHT and event.pressed == true:
			pass # TODO clear terrain from target location

			
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed == true:
			if m_hexes.has(hexpt):
				match state:
					BASE:
						if m_hexes[hexpt] != null:
							# pickup again
							var tile_data = m_hexes[hexpt]
							drag_sprite = tile_data[0]
							drag_data = tile_data[1]
							hexmap.remove_child(drag_sprite)
							add_child(drag_sprite)
							drag_sprite.set_position(event.position)
							m_hexes[hexpt] = null
							state = DRAG_UNIT
					PLACE_TERRAIN:
						var idx = terrain_option.get_selected()
						if idx > 0:
							generate_terrain(cube, idx)
							terrain_type[hexpt] = idx
					
					
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed == false:
			match state:
				DRAG_UNIT:
					if m_hexes.has(hexpt):
						remove_child(drag_sprite)
						hexmap.add_child(drag_sprite)
						drag_sprite.set_position(hexmap.point_transform * hexpt)
						var team_name = "player" if grid_coord(hexpt).x < 4 else "enemy"
						drag_sprite.flip_h = team_name == "enemy"
						m_hexes[hexpt] = [drag_sprite, drag_data, team_name]
					else:
						drag_sprite.queue_free()
						
					drag_sprite = null 
					drag_data = null
					state = BASE

				
func connect_unit_panel(panel):
	panel.pickup_unit.connect(_on_pickup_unit)

func _on_pickup_unit(data):
	# create a temporary sprite to represent the dragging
	drag_sprite = Sprite2D.new()
	drag_sprite.texture = load(data['sprite_path'])
	add_child(drag_sprite)
	drag_data = data
	state = DRAG_UNIT

## terrain generation
				
func generate_terrain(cube, idx):
	match idx:
		1:
			pass # plains
		2:
			generate_forest(cube, rng.randi_range(20,35))
		3:
			generate_hill(cube)
		4:
			generate_swamp(cube)
		5:
			generate_impassable(cube)
			

func generate_forest(cube, n_tree):
	# pass a reference to hexmap to better insulate the random generation code
	var pts = background.random_points_in_hex(cube, hexmap, n_tree)
	background.generate_forest(pts)

func generate_hill(cube):
	var pos = hexmap.to_global(hexmap.cube_to_pixel(cube)) + 0.1 * tile_size * Vector2(rng.randfn(), rng.randfn())
	background.generate_hill(pos)
	
func generate_swamp(cube):
	var pos = hexmap.to_global(hexmap.cube_to_pixel(cube))
	background.generate_swamp(pos)
	
func generate_impassable(cube):
	var pos = hexmap.to_global(hexmap.cube_to_pixel(cube)) + 0.02 * tile_size * Vector2(rng.randfn(), rng.randfn())
	background.generate_impassable(pos)

	
## save terrain

func get_terrain_resource():
	var terrain_type_data = []
	for k in terrain_type.keys():
		terrain_type_data.append([k, terrain_type[k]])
	var struct = TerrainResource.new()
	struct.type_data = terrain_type_data
	return struct

func save_terrain():
	var struct = get_terrain_resource()
	var save_target = "res://tools/scene/save_resource/" + "map_terrain.tres"
	Message.push("save to %s" % save_target)
	print(struct.type_data)
	ResourceSaver.save(struct, save_target)

func get_layout_resource():
	var struct = LayoutResource.new()
	for hexpt in m_hexes:
		var tile_data = m_hexes[hexpt]
		if tile_data == null: continue
		if tile_data[2] == "player":
			struct.left_team.append([hexpt, tile_data[1]])
		elif tile_data[2] == "enemy":
			struct.right_team.append([hexpt, tile_data[1]])
	return struct

func save_layout():
	var struct = get_layout_resource()
	var save_target = "res://tools/scene/save_resource/" + "map_layout.tres"
	Message.push("save to %s" % save_target)
	ResourceSaver.save(struct, save_target)
	
