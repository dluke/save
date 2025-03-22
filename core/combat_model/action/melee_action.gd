extends Action
class_name MeleeAction

var other

func _init(_token, _other):
	super(_token)
	other = _other

func capable():
	return token.can_attack(other)

func forward():
	token.attack(other)
	
func play():
	await token.animate_attack(other) # note. triggers counter

func logstr():
	return "%s melee attack against %s" % [str(token), str(other)]
