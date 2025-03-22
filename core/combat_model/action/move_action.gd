extends Action

class_name MoveAction

var path
var mv_cost

func _init(_token, _path, _mv_cost):
	super(_token)
	path = _path
	mv_cost = _mv_cost


func forward():
	self.token.move_path(path, mv_cost)

func play():
	await token.animate_move_path(path)

func _to_string():
	return "MoveAction(%s %s -> %s)" % [str(token), path[-1], path[0]]

func logstr():
	return "%s move %s -> %s" % [str(token), path[-1], path[0]]
