extends Action
class_name CounterAction

var other

func _init(_token, _other):
	super(_token)
	other = _other
	
func capable():
	return token.can_counter(other)
	
func forward():
	token.counter(other)
	
func play():
	await token.animate_counter(other) 

func logstr():
	return "%s counter attack against %s" % [str(token), str(other)]
