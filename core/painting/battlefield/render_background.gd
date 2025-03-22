extends Node2D

##
# background sprites
@onready var nature_layer = $nature

# for painting background textures
@onready var draw_surface = $SubViewport/draw_surface
@onready var stamp_node = $stamp
var stamp : Image

# loading assets
var nature_resource_path = "res://assets/placeholder/nature/"
var tree_file = [
	"Fir_tree_1.png",
	"Fir_tree_2.png",
	"Fir_tree_3.png",
	"Fir_tree_4.png",
	"Fir_tree_5.png",
	"Dry_tree_5.png"
]
var tree_texture = []

var hill_file = [
	"Hill01.png",
	"Hill02.png",
	"Hill03.png"
]
var hill_texture = []

var impassable_file = [
	"Mount01.png",
	"Mount02.png",
#	"Mount03.png"
]
var impassable_texture = []

# 
var rng = RandomNumberGenerator.new()


func _ready():
	for filename in tree_file:
		tree_texture.append( load(nature_resource_path + filename) )
	for filename in hill_file:
		hill_texture.append( load(nature_resource_path + filename) )
	for filename in impassable_file:
		impassable_texture.append( load(nature_resource_path + filename) )

	# background texture
	stamp = stamp_node.get_texture().get_image()
	var canvas_size = draw_surface.get_size()
	var blank = Image.create(canvas_size.x, canvas_size.y, true, Image.FORMAT_RGBA8)
	blank.fill(Color(0,0,0,0));
	draw_surface.texture = ImageTexture.create_from_image(blank)

func generate_tree(pos):
	var random_index = rng.randi() % tree_texture.size()
	var sprite = Sprite2D.new()
	sprite.set_texture(tree_texture[random_index])
	sprite.set_position(pos)
	nature_layer.add_child(sprite)

func random_points_in_hex(cube, hexmap, n):
	var pts = []
	for i in range(n):
		pts.append(hexmap.to_global(random_point(cube, hexmap)))
	return pts
	
func random_point(cube, hexmap):
	var tile_size = hexmap.hex_size
	var contained_pt = Vector2(INF, INF)
	for i in range(100): # max tries
		contained_pt = Vector2(rng.randi_range(0, tile_size[0]), rng.randi_range(0, tile_size[1]))
		if hexmap.pixel_to_cube(contained_pt).get_axial() == Vector2(0,0):
			break
	if contained_pt.x == INF:
		print("warning random point generation failed")
	# generate random position in hex
	var pt = hexmap.cube_to_pixel(cube) + contained_pt - tile_size/2
	return pt

func generate_forest(pos_array):
	for pos in pos_array:
		generate_tree(pos)

func generate_hill(pos):
	var random_index = rng.randi() % hill_texture.size()
	var sprite = Sprite2D.new()
	sprite.set_texture(hill_texture[random_index])
	sprite.set_position(pos)
	sprite.set_scale(Vector2(2,2))
	nature_layer.add_child(sprite)
	
func generate_swamp(pos):
	var mask_image = draw_surface.texture.get_image() # can avoid copy?
	mask_image.blend_rect(stamp, Rect2i(Vector2i(), stamp.get_size()), Vector2i(pos) - stamp.get_size()/2)
	draw_surface.texture.update(mask_image)
	
func generate_impassable(pos):
	var random_index = rng.randi() % impassable_texture.size()
	var sprite = Sprite2D.new()
	sprite.set_texture(impassable_texture[random_index])
	sprite.set_position(pos)
	sprite.set_scale(Vector2(0.8,0.8))
	nature_layer.add_child(sprite)

