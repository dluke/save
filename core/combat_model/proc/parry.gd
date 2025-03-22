extends Proc

# preload
var FloatText = preload("res://core/token/float_text.tscn")

var mtoken
var skill_value
var indicator = null
var parry_available = true

func _init(mt : TokenResource, value):
	mtoken = mt
	skill_value = value
	enum_list = [
		TokenResource.Proc.UPKEEP,
		TokenResource.Proc.PRE_HIT,
		TokenResource.Proc.POST_HIT
	]
	
#func setup():
#	indicator = FloatText.instantiate()
#	indicator.set_visible(false)
#	indicator.set_text("Parry")
#	token.add_child(indicator)

		
func forward(proc):
	match proc:
		TokenResource.Proc.UPKEEP:
			parry_available = true
		TokenResource.Proc.PRE_HIT:
			mtoken.modify_bonus_stat(TokenResource.Stats.Defense, skill_value)
			parry_available = false
		TokenResource.Proc.POST_HIT:
			mtoken.modify_bonus_stat(TokenResource.Stats.Defense, -skill_value)

#func play(proc):
#	match proc:
#		TokenResource.Proc.PRE_HIT:
#			indicator.set_visible(true)
#			# tween base float text animation (TODO. Animation System)
#			var size_x = indicator.get_size().x
#			indicator.set_position(Vector2(-size_x/2, -80))
#			var tween = indicator.create_tween()
#			var final_position = indicator.get_position() + Vector2(0,-50)
#			tween.tween_property(indicator, "position", final_position, 0.4)
#			tween.parallel().tween_property(indicator, "modulate", Color(1,1,1,0), 0.3)
#			await indicator.get_tree().create_timer(0.4).timeout
#			indicator.set_visible(false)
