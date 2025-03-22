extends Proc

# preload
var ragebuff = preload("res://assets/scene/ragebuff.tscn")

var mtoken
var skill_value
var health_threshold = 0.75
var indicator = null

func _init(mt : TokenResource, value):
	mtoken = mt
	skill_value = value
	enum_list = [
		TokenResource.Proc.TAKE_DAMAGE
	]
	
#func setup():
#	indicator = ragebuff.instantiate()
#	indicator.set_visible(false)
#	token.sprite.add_child(indicator)
	
func condition():
	# only activate if enough damage is taken
	var Health = mtoken.Stats.Health
	return mtoken.get_stat(Health)/mtoken.get_base_stat(Health) < health_threshold
	
func forward(proc):
	match proc:
		TokenResource.Proc.TAKE_DAMAGE:
			if condition():
				mtoken.modify_bonus_stat(mtoken.Stats.Attack, skill_value)


#func play(proc):
#	match proc:
#		TokenResource.Proc.TAKE_DAMAGE:
#			if condition():
#				indicator.set_visible(true)
