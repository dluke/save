extends Resource
class_name SpellAnimationResource

@export var animation_name : String
@export var sprite_frames : SpriteFrames = load("res://assets/placeholder/animation/resource/spell_sprite.tres")

func play_on_token(token):
	var animated_sprite = AnimatedSprite2D.new()
	animated_sprite.set_sprite_frames(sprite_frames)
	token.add_child(animated_sprite)
	animated_sprite.play(animation_name)
	await animated_sprite.animation_looped
	animated_sprite.queue_free()
