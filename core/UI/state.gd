extends Node
class_name State

var fsm: StateMachine
var node: Node # the node that the the input state machine is attached to

func _ready():
	fsm = get_parent()
	node = fsm.get_parent()

func enter():
	pass

func exit():
	pass

func input(_event):
	pass 

func unhandled_input(_event):
	pass 
