extends State
class_name SharedState

# Notes
# the parent class of selected and grabbed states

## setup

@onready var Cursor = %Cursor
@onready var Message = %Message

var token
var m_viable_tiles = []
var selected_coord = null

## enter/exit

func enter():
	pass

	
func exit():
	pass
	
## handle input

func input(event):
	pass
	
func _shared_enter():
	var mvpt = token.m_resource.mvpt
	
	var tiles = token.pathing.hexmap.get_tile_range(token.tile, mvpt+1)
	token.pathing.clear()
	token.pathing.dijkstra(token.tile, tiles, token.m_resource.movecost)
	
	# Note. more efficient to copy over pathing data from grabbed state
	m_viable_tiles = []
	if token.capable_move(): 
		m_viable_tiles = token.pathing.viable(token)
	
	if token.capable_ranged_attack():
		var tile_set = tiles_in_range(m_viable_tiles, token.get_stat(TokenResource.Stats.Range))
		for unit in token.get_team().get_active_unit_list():
			tile_set.erase(unit.tile)
		node.show_aim_range(tile_set)

	node.show_movement_range(m_viable_tiles)
	
	selected_coord = node.local_to_map(node.cube_to_pixel(token.tile.cube))
	node.set_indicator(selected_coord, node.ATLAS_SELECT)
	
func tiles_in_range(viable_tiles, rg):
	var tiles = {}
	for tile in viable_tiles:
		for t in node.get_tile_range(tile, rg, 2):
			tiles[t] = 0
	return tiles 
	
# utilities
func get_directional_path(other, local_position):
	# find the path that moves the hex closest to the cursor
	var path = null # return null if there were no viable tiles to move
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



