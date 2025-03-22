extends Resource
class_name SpellResource

@export var name : String
@export var texture : Texture2D
@export var gem_cost : int
@export var stamina_cost : int
@export var target_condition : Condition
@export var animation : SpellAnimationResource

enum Condition {ALL, FRIEND, ENEMY}

var target : Token

func set_target(_t):
	target = _t
	
func can_target(token):
	var result = false
	match target_condition:
		Condition.ALL:
			result = true
		Condition.FRIEND:
			result = token.get_team().player_control
		Condition.ENEMY:
			result = not token.get_team().player_control
	return result

func forward():
	pass
	
func play():
	animation.play_on_token(target)

func logstr():
	return ""

func _to_string():
	return name
