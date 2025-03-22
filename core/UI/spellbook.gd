extends PanelContainer

var Spellslot = preload("res://core/UI/spellslot.tscn")

signal spell_selected

@onready var grid = $margin/grid

## setup

func _ready():
	for _i in range(3*7):
		var slot = Spellslot.instantiate()
		slot.spell_selected.connect(_on_spell_selected)
		grid.add_child(slot)

func set_spells(spell_list):
	var slot_list = grid.get_children()
	for i in range(spell_list.size()):
		var spell = spell_list[i]
		var slot = slot_list[i]
		slot.set_spell(spell)

## implement
func _on_spell_selected(spell):
	spell_selected.emit(spell)
