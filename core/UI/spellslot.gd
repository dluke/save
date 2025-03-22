extends PanelContainer

# primary component of spellbook

signal spell_selected

#
@export var spell : SpellResource

# component
@onready var texture = $texture

# represent combat state
var used = false

func set_spell(_spell):
	spell = _spell
	texture.set_texture(spell.texture)

func _gui_input(event):
	if spell == null:
		return

	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.is_pressed():
			spell_selected.emit(spell)
