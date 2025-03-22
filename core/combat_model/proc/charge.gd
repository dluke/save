extends Proc

# preload
var StatusPair = preload("res://core/token/status_pair.tscn")

var mtoken
var skill_value
var up_value = 0
var indicator = null

func _init(mt : TokenResource, value):
	mtoken = mt
	skill_value = value
	enum_list = [
		TokenResource.Proc.MOVE,
		TokenResource.Proc.MELEE,
		TokenResource.Proc.DOWNKEEP
	]
	
# TODO remove all references to Token from here (token_resource is OK)
	
#func setup():
#	indicator = StatusPair.instantiate()
#	indicator.set_visible(false)
#	mtoken.status_box.add_child(indicator)
	
func forward(proc):
	match proc:
		TokenResource.Proc.MOVE:
			var travel = mtoken.tile.cube.distance(mtoken.origin_tile.cube)
			up_value = skill_value * travel
			mtoken.modify_bonus_stat(TokenResource.Stats.Attack, up_value)
		TokenResource.Proc.MELEE, TokenResource.Proc.DOWNKEEP:
			mtoken.modify_bonus_stat(TokenResource.Stats.Attack, -up_value)
			up_value = 0

#func play(proc):
#	match proc:
#		TokenResource.Proc.MOVE:
#			var travel = token.tile.cube.distance(token.origin_tile.cube)
#			indicator.set_text('+' + str(skill_value * travel))
#			indicator.set_visible(true)
#		TokenResource.Proc.MELEE, TokenResource.Proc.DOWNKEEP:
#			indicator.set_visible(false)
