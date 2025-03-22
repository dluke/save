extends TileMap
class_name HexTileMap

# for hexgrid coordinates
var hex_direction_vector = [
	Cube.new(+1, 0, -1), Cube.new(0, +1, -1), Cube.new(-1, +1, 0),
	Cube.new(-1, 0, +1), Cube.new(0, -1, +1), Cube.new(+1, -1, 0),
]

var hex_size = Vector2(tile_set.get_source(0).get_texture_region_size()) # Vector2(104, 120);
var point_transform = Transform2D(
	hex_size[0] * Vector2(1, 0),
	hex_size[0] * Vector2(0.5, 0.5 * sqrt(3)),
	hex_size/2 # origin
	)
var hex_transform = point_transform.affine_inverse()

## Cube coordinates

func cube_to_pixel(cube):
	return point_transform * cube.get_axial()
	
func pixel_to_cube(point):
	var hex = hex_transform * point
	return Cube.nearest(hex[0], hex[1])
	
func get_range(cube, n, min_distance=0):
	var lst = [cube]
	if n == 0:
		return lst
	for q in range(-n, n+1):
		for r in range(max(-n, -q-n), min(n, -q+n)+1):
			var s = -q-r
			var candidate = cube.add(Cube.new(q,r,s))
			if candidate.distance(cube) >= min_distance:
				lst.append( candidate )
	return lst

## cube

class Cube:
	
	var q: int
	var r: int
	var s: int
	
	func _init(_q, _r, _s):
		q = _q
		r = _r
		s = _s
		
	func get_axial():
		return Vector2(q, r)
	
	func add(other: Cube):
		return Cube.new(q + other.q, r + other.r, s + other.s)
		
	func subtract(other: Cube):
		return Cube.new(q - other.q, r - other.r, s - other.s)
		
	func distance(other):
		var vec = subtract(other)
		return (abs(vec.q) + abs(vec.r) + abs(vec.s)) / 2
		
	static func nearest(x, y): # x,y are decimal axial coordinates
		var _x = round(x)
		var _y = round(y)
		x -= _x; y -= _y
		if abs(x) >= abs(y):
			var _q = _x + round(x + 0.5*y)
			return Cube.new(_q, _y, -_y-_q)
		else:
			var _r = _y + round(y + 0.5*x)
			return Cube.new(_x, _r, -_x-_r)
	
	func _to_string():
		return str(Vector3(q, r, s))

