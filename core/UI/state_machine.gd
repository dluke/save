extends Node
class_name StateMachine

# Notes 
# state machine with history

##

var state: State
var history = []

# states
@onready var base = $base
@onready var view = $view
@onready var grabbed = $grabbed
@onready var selected = $selected
@onready var spell = $spell
@onready var inspect = $inspect

## 

func _ready():
	enter_base_state()

func enter_base_state():
	# Set the initial state to the first child node
	state = get_child(0)
	history.append(state)
	_enter_state()
	
func to_state(new_state):
	state.exit()
	state = new_state
	history.append(state)
	_enter_state()
	
func exit():
	state.exit()
	history.append(state)
	state = null
	
func enter_state(new_state):
	state = new_state
	_enter_state()

func back(): 
	if history.size() > 0:
		state.exit()
		var from_state = history.pop_back()
		state = history[-1]
		_enter_state()
		return from_state
	else:
		pass
#		Log.warn('Failed to revert statemachine, no history')

func _enter_state():
#	print('enter state ', state.get_name()
	state.enter()

# Route Game Loop function calls to
# current state handler method if it exists
# func _process(delta):
# 	if state.has_method("process"):
# 		state.process(delta)

func _input(event):
	if is_instance_valid(state) && state.has_method("input"):
		state.input(event)

func _unhandled_input(event):
	if is_instance_valid(state) && state.has_method("unhandled_input"):
		state.unhandled_input(event)

func _unhandled_key_input(event):
	if is_instance_valid(state) && state.has_method("unhandled_key_input"):
		state.unhandled_key_input(event)

func handle_signal(signal_, args=[]):
	if is_instance_valid(state) && state.has_method(signal_):
		state.call(signal_, args)

# func _notification(what):
# 	if state && state.has_method("notification"):
# 		state.notification(what)

# debugging

func _print_history():
	var hs = []
	for st in history:
		hs.append(st.name)
	return hs
