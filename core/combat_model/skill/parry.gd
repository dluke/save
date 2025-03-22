extends SkillResource

var proc = preload("res://core/combat_model/proc/parry.gd")

func gain(mtoken):	
	mtoken.add_trigger(proc.new(mtoken, self.value))
