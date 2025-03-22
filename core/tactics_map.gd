extends Node2D

var TacticsCard = preload("res://core/UI/tactics_card.tscn")

## members
@onready var Message = %Message

@onready var ulayer = %unit_layer
@onready var hexmap = %hexmap
@onready var background = $background
@onready var spellbook = %Spellbook
@onready var lbox = %LeftCardBox
@onready var rbox = %RightCardBox

#
@export var terrain : TerrainResource
@export var layout : LayoutResource


var player_team
var enemy_team

# turn system
var active_team = null
var is_combat_active = false


## setup

func make_unique_name(cls_name, team):
	var unique_name = "%s_%s" % [team.name, cls_name]
	var regex = RegEx.new()
	regex.compile("\\d+")
	for unit in team.get_unit_list():
		if unit.get_unique_name() == unique_name:
			var result = regex.search(unique_name)
			if result:
				var s = result.get_string()
				unique_name = unique_name.slice(0, -s.length()) + str(int(s)+1)
			else:
				unique_name = unique_name + "2"
	return unique_name


func _ready():
	
	# essential references
	connect_hexmap()

	# terrain
	for item in terrain.type_data:
		var hex = item[0]
		var terrain_idx = item[1]
		# set terrain
		hexmap.tiles[hex].terrain = terrain_idx
		# render background
		var cube = HexTileMap.Cube.new(hex[0], hex[1], -hex[0]-hex[1])
		paint_terrain(cube, terrain_idx)
	
	# essential combat setup
	player_team = Team.new('player')
	enemy_team  = Team.new('enemy')
	enemy_team.m_opposed = player_team
	player_team.m_opposed = enemy_team
	enemy_team.facing = Team.Facing.LEFT
	enemy_team.player_control = false
	player_team.initiative = 1
	
	# spells
	var spell_list = [
		load("res://core/combat_model/spells/magic_armor.tres")
	]
	spellbook.set_spells(spell_list)
	spellbook.spell_selected.connect(_on_spell_selected)

	# assemble the layout
	var teams = [player_team, enemy_team]
	var layout_pair = [layout.left_team, layout.right_team]
	for team_idx in range(2):
		var team = teams[team_idx]
		for item in layout_pair[team_idx]:
			var hex = item[0]
			var unit_data = item[1]
			var unique_name = make_unique_name(unit_data["class"], team)
			# use stored unit data
#			var token = TemplateConstructor.construct(unit_data, unique_name)
			# construct unit from base class
			var token = TemplateConstructor.new_class_unit(unit_data["class"], unique_name)
			team.add_unit(token)
			hexmap.add_token(token)
			hexmap.set_grid_position(token, hex)
	
	# tmp. pick an enemy and connect tacticscard 
	for token in player_team.get_unit_list():
		var card = TacticsCard.instantiate()
		lbox.add_child(card)
		card.connect_token(token)
	
	for token in enemy_team.get_unit_list():
		var card = TacticsCard.instantiate()
		rbox.add_child(card)
		card.connect_token(token)
	

	# ...
	
	init_setup_phase()
	
	# and then trigger init_combat later
	
	init_combat()

	# testing
#	hexmap.m_selected_item = p1
#	var fsm = $grid_layer/hex_tile_map/InputStateMachine
#	fsm.to_state(fsm.selected)

# render terrain

func save_token(token):
	var save_path = "res://tmp/save_state/%s.tres" % token.get_unique_name()
	print('save to ', save_path)
	ResourceSaver.save(token.m_resource, save_path)
	
func load_token(name):
	var load_path = "res://tmp/save_state/%s.tres" % name
	var tkr = ResourceLoader.load(load_path)
	var token = TemplateConstructor.Token.instantiate()
	token.m_resource = tkr
	token.m_name = tkr.name
	token.assemble_triggers()
	return token
	

func paint_terrain(cube, idx):
	match idx:
		1:
			generate_forest(cube, background.rng.randi_range(20,35))
		2:
			generate_hill(cube)
		3:
			generate_swamp(cube)
		4:
			generate_impassable(cube)

func generate_forest(cube, n_tree):
	# pass a reference to hexmap to better insulate the random generation code
	var pts = background.random_points_in_hex(cube, hexmap, n_tree)
	background.generate_forest(pts)

func generate_hill(cube):
	var rng = background.rng
	var pos = hexmap.to_global(hexmap.cube_to_pixel(cube)) + 0.1 * hexmap.hex_size * Vector2(rng.randfn(), rng.randfn())
	background.generate_hill(pos)
	
func generate_swamp(cube):
	var pos = hexmap.to_global(hexmap.cube_to_pixel(cube))
	background.generate_swamp(pos)
	
func generate_impassable(cube):
	var rng = background.rng
	var pos = hexmap.to_global(hexmap.cube_to_pixel(cube)) + 0.02 * hexmap.hex_size * Vector2(rng.randfn(), rng.randfn())
	background.generate_impassable(pos)


# setup signals

func connect_hexmap(): 
	hexmap.Model = self
	hexmap.end_turn.connect(_on_end_turn)
	hexmap.end_combat.connect(_on_end_combat)

func _on_end_turn():
	end_turn()
	
func _on_end_combat():
	is_combat_active = false
	hexmap.fsm.to_state(hexmap.fsm.view)
	
func _on_spell_selected(spell):
	toggle_spellbook()
	var spell_state = hexmap.fsm.spell
	spell_state.spell = spell
	hexmap.fsm.to_state(spell_state)
	
	
## simple combat logic
# 1. team with the highest initiative goes first
# 2. uses up movement points, refreshed at turn change
# 3. turn change button
# 4. update step

func init_setup_phase():
	pass

func init_combat():
	is_combat_active = true
	active_team = player_team if player_team.initiative > enemy_team.initiative else enemy_team
	_update_input_state()
	Message.push("%s turn" % active_team.name)
	upkeep()
	for token in active_team.get_opposed().m_unit_list:
		token.upkeep() # upkeep the other team as will
	
func next_team():
	active_team = enemy_team if active_team == player_team else player_team
	_update_input_state()
	
func _update_input_state():
	if active_team.player_control:
		hexmap.fsm.to_state(hexmap.fsm.base)
	else:
		hexmap.fsm.to_state(hexmap.fsm.view)

func upkeep():
	for token in active_team.m_unit_list:
		if token.is_alive():
			token.upkeep()
			# snap the tokens on upkeep in case they moved as part of animations (breaks model/view patter)
			hexmap.snap(token, token.tile)
		
func downkeep():
	for token in active_team.m_unit_list:
		if token.is_alive():
			token.downkeep()

func end_turn():
	downkeep()
	next_team()
	Message.push("%s turn" % active_team.name)
	upkeep()
	if active_team.player_control == false:
		ai_behaviour()
		
func ai_behaviour():
	# play AI moves
	for token in active_team.m_unit_list:
		if !token.is_alive():
			continue
		var action_list = token.evaluate()
		for action in action_list:
			if action.capable():
				print('LOG: ', action.logstr())
				action.forward()
				await action.play()
	if is_combat_active:
		end_turn()
	
## input

func _unhandled_key_input(event):
	if event is InputEventKey and event.pressed == true and event.keycode == KEY_S:
		toggle_spellbook()

func toggle_spellbook():
	spellbook.set_visible(!spellbook.visible)



## inner class
	
class Team:
	
	enum Facing {RIGHT, LEFT}

	var name 
	var player_control = true
	var m_unit_list = []
	var m_opposed = null 
	var facing = Team.Facing.RIGHT
	var initiative = 0
	
	func _init(_name):
		name = _name
		
	func get_opposed(): # return the opposing team
		return m_opposed
	
	func get_unit_list():
		return m_unit_list

	func get_active_unit_list():
		var lst = []	
		for unit in m_unit_list:
			if unit.is_alive():
				lst.append(unit)
		return lst
		
	func add_unit(unit):
		m_unit_list.append(unit)
		unit.set_team(self)
		
	func _to_string():
		return "Team(%s)" % name
	
	
	
