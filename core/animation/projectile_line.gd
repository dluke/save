extends Line2D
class_name ProjectileLine

var m_end_position = null
var m_duration = 5
var m_frame = 0
var lvl = [0.1, 0.6, 1.0, 1.0]

func _init(start_position, end_position):
	m_end_position = end_position 
	add_point(start_position)
	add_point(start_position)
	# style
	width = 2
	antialiased = true
	default_color = Color(0.65, 0.65, 0.65, 0.3)

func _process(delta):
	var idx = min(m_frame, lvl.size()-1)
	var var_position = get_point_position(0) + lvl[idx] * (m_end_position - get_point_position(0)) 
	set_point_position(1, var_position)
	m_frame += 1
	if m_frame > m_duration:
		queue_free()
	
	
