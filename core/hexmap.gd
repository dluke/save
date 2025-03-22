extends HexTileMap
class_name HexMap

## signals

signal end_turn
signal end_combat

## members

# ready by parent
var Model = null 

# outside sub-tree
@onready var Cursor = %Cursor
@onready var Message = %Message

# inside subtree
@onready var fsm = $InputStateMachine
@onready var unit_layer = %unit_layer
@onready var glyph_layer = %glyph_layer
@onready var animation_layer = %animation_layer

var tiles: Dictionary # axial coordinates as keys

# tilemap display constants
const ATLAS_SELECT = Vector2i(2, 0)
const ATLAS_HIGHLIGHT = Vector2i(3, 0)
const ATLAS_AIM = Vector2i(3, 2)
var info_layer = 1

# dragging (use input state machine)
var m_drag_item = null
var m_selected_item = null
var m_current_movement = [] # list of tiles that can be moved to
var m_current_aim = [] # list of tiles that can be moved to

# glyphs
var glyph_list = []
enum Glyph {DOWNARROW}
const glyph_texture = [
	preload("res://assets/placeholder/icon/cursor/arrow_s.png")
]

## setup

func _ready():
	# populate tiles in 8x8 grid
	_populate()
	
func _populate():
	var left = Cube.new(0, 0, 0)
	var east = hex_direction_vector[0]
	var down = [hex_direction_vector[1], hex_direction_vector[2]]
	# tiles
	for i in range(8):
		var current = left
		for j in range(8):
			var tile = Tile.new()
			tile.cube = current
			tiles[current.get_axial()] = tile
			current = current.add(east)
		left = left.add(down[i % 2])
	# adjacency
	for axial in tiles:
		var tile = tiles[axial]
		for direction in hex_direction_vector:
			var cube_adj = tile.cube.add(direction)
			var tile_adj = tiles.get(cube_adj.get_axial()) # null for boundary
			tile.adjacent.append(tile_adj)
	
func add_token(token):
	unit_layer.add_child(token)
	token.kill.connect(_on_kill)
	token.connect_hexmap(self)

func remove_token(token):
	unit_layer.remove_child(token)

## implement signals

func _on_kill(token):
	# clear references
	token.tile.slot = null
#	token.tile = null keep reference to the tile it died on
	# check win condition
	var any_alive = false
	for item in Model.active_team.get_unit_list():
		if item.is_alive():
			any_alive = true
	if !any_alive:
		Message.push("Team %s wins" % token.get_team().get_opposed().name)
		end_combat.emit()
	# render death animation
	await animate_kill(token) # TODO decouple kill animaton from removing child?
	remove_token(token)
	
func animate_kill(token):
	token.animate.play("fade")
	await token.animate.animation_finished



## pickup / drop / select

func position_to_hexpt(pos): # for input handling
	return pixel_to_cube(to_local(pos)).get_axial()

func snap(item, tile): 
	item.set_hex_position(point_transform * tile.cube.get_axial())

func move_path(unit, path, mv_cost):
	unit.move_path(path, mv_cost) # combat model
	snap(unit, path[0])
	
func set_grid_position(item, qr: Vector2):
	var tile = tiles[qr]
	tile.fill_slot(item)
	snap(item, tile)

func pickup(item):
#	Message.push('pick up %s at %s' % [str(item), str(item.tile)] )
	m_drag_item = item
	Cursor.set_texture(Cursor.HANDCLOSED)

func drop_at(tile):
	snap(m_drag_item, tile) # updates tile/slot references
	Cursor.set_texture(Cursor.HANDOPEN)

## visually modify tiles

func set_indicator(coord, atlas_coord):
#	var coord = local_to_map(local_position)
	set_cell(info_layer, coord, 0, atlas_coord)
	
func clear_indicator(coord):
	set_cell(info_layer, coord, -1)
		
func show_movement_range(tile_lst):
	for tile in tile_lst:
		var coord = local_to_map(cube_to_pixel(tile.cube)) # tilemap coord
		set_indicator(coord, ATLAS_HIGHLIGHT)
		m_current_movement.append(tile)

func clear_movement_range():
	for tile in m_current_movement:
		clear_indicator(local_to_map(cube_to_pixel(tile.cube)))
	m_current_movement.clear()
	
func show_aim_range(tile_list):
	for tile in tile_list:
		var coord = local_to_map(cube_to_pixel(tile.cube)) # tilemap coord
		set_indicator(coord, ATLAS_AIM)
		m_current_aim.append(tile)
		
func clear_aim_range():
	for tile in m_current_aim:
		clear_indicator(local_to_map(cube_to_pixel(tile.cube)))
	m_current_aim.clear()

func add_glyph(tile, x):
	var glyph = Sprite2D.new()
	glyph.set_texture(glyph_texture[x])
	glyph.set_position(cube_to_pixel(tile.cube))
	$glyph_layer.add_child(glyph)
	glyph_list.append(glyph)

func clear_glyphs():
	for glyph in glyph_list:
		glyph.queue_free()
	glyph_list.clear()
	
func draw_path(path):
	# draw path using line2D
	var line_path = Node2D.new()
	for i in range(path.size()-1):
		var p1 = cube_to_pixel(path[i].cube)
		var p2 = cube_to_pixel(path[i+1].cube)
		var line = Line2D.new()
		line.add_point(p1.lerp(p2, 0.1))
		line.add_point(p1.lerp(p2, 0.9))
		line.width = 1
		line.default_color = Color(1,1,1,0.8)
		line_path.add_child(line)
	glyph_layer.add_child(line_path)
	
func get_tile_range(tile, n, min_distance=0):
	# iterate cube coordinates and keep only tiles in the map
	var lst = get_range(tile.cube, n, min_distance)
	var tile_lst = [] # include the current tile
	for item in lst:
		var hexpt = item.get_axial()
		if hexpt in tiles:
			tile_lst.append(tiles[hexpt])
	return tile_lst


