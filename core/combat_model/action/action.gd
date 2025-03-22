extends RefCounted

class_name Action

# base class for actions that tokens can make

var token

func _init(_token):
	token = _token

func capable():
	return true
	
func forward():
	pass # update model

func play():
	pass # animation

func logstr():
	return ""
