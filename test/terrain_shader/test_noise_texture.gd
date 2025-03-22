extends Node2D

@onready var draw_surface = $SubViewport/draw_surface
@onready var stamp_node = $stamp
var stamp : Image

func _ready():
	stamp = stamp_node.get_texture().get_image()
	var canvas_size = draw_surface.get_size()
	var blank = Image.create(canvas_size.x, canvas_size.y, true, Image.FORMAT_RGBA8)
	blank.fill(Color(0,0,0,0));
	draw_surface.texture = ImageTexture.create_from_image(blank)


func _input(event):
	if event is InputEventMouseButton and event.pressed:
		var image = draw_surface.texture.get_image() # can avoid copy?
		image.blend_rect(stamp, Rect2i(Vector2i(), stamp.get_size()), Vector2i(event.position) - stamp.get_size()/2)
		draw_surface.texture.update(image)

	
