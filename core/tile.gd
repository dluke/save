extends Resource
class_name Tile

# create a tile for every battlefield hex and have it store data?
# can store tiles in a dictionary using axial coordinates
enum Terrain {PLAINS, FOREST, HILL, SWAMP, IMPASSABLE}
const DEF_MOVECOST = [1, 2, 3, 3, INF]

var cube: HexTileMap.Cube
var slot: Node
var terrain = Terrain.PLAINS 
var adjacent = [] # null for boundaries

func get_default_movement_cost():
	return DEF_MOVECOST[terrain] # INF means impassable

func clear_slot():
	slot = null

func fill_slot(token):
	slot = token
	token.m_resource.tile = self 
	
func is_empty():
	return slot == null

func _to_string():
	return 'Tile%s' % str(cube.get_axial())
