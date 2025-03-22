extends Action
class_name RangedAction

var other

func _init(_token, _other):
	super(_token)
	other = _other
	
func capable():
	return token.can_shoot(other)

func forward():
	token.ranged_attack(other)
	
func play():
	await token.animate_ranged_attack(other) 

func logstr():
	return "%s ranged attack against %s" % [str(token), str(other)]
