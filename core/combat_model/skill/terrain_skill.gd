extends SkillResource
class_name TerrainSkill

@export var terrain_type : Tile.Terrain

func gain(model):
	model.movecost[terrain_type] = 1
	model.terrain_stamina_cost[terrain_type] = 0
	
func lose(model):
	model.movecost[terrain_type] = Tile.DEF_MOVECOST[terrain_type]
	model.terrain_stamina_cost[terrain_type] = 1
