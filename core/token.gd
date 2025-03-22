extends Node2D
class_name Token


var FloatText = preload("res://core/token/float_text.tscn")

##
signal kill
signal stat_changed

## members
@onready var pathing = $pathing
@onready var stacking_hbox = $UILayer/stacking_hbox

# from parent (tmporary)
var hexmap 
func connect_hexmap(_hexmap):
	hexmap = _hexmap # tmp
	pathing.hexmap = hexmap

#
var m_name # todo clean up display and unique name references (both belong to the combat model?)
var tile : Tile:
	get:
		return m_resource.tile

var m_facing = 0
var m_team = null
@export var m_resource : TokenResource:
	set(x):
		m_resource = x
		m_name = m_resource.name
		
func get_display_name():
	return m_name
	
func get_unique_name():
	return m_resource.name

# battlefield state (should be part of m_resource?, or another combat model component?)
var acted = false
var moved = false


# components
@onready var sprite = $sprite
@onready var health_bar = %HealthBar
@onready var stamina_bar = %StaminaBar
@onready var morale_bar = %MoraleBar
@onready var animate = %animate
@onready var status_box = %StatusBox

# save position in the grid so that we can offset the token easily
var hex_position
func set_hex_position(pos):
	hex_position = pos
	position = hex_position

# step the sprite towards an enemy and back
const grid_distance = 104 # todo retrieve this from somewhere
var melee_animation_vector = grid_distance * Vector2(1,0)
@export var melee_animation_var: float = 0.0:
	set(value):
		melee_animation_var = value
		set_position(hex_position + melee_animation_var * melee_animation_vector)

# same for movement?
var move_animation_path = []
@export var move_animation_var: float = 0.0:
	set(value):
		if move_animation_path.is_empty():
			return
		move_animation_var = value
		var n = move_animation_path.size()-1
		var start_idx = int(move_animation_var / (1.0/n))
		if start_idx == n:
			set_position(hexmap.cube_to_pixel(move_animation_path[n].cube))
		else:
			var p1 = hexmap.cube_to_pixel(move_animation_path[start_idx].cube)
			var p2 = hexmap.cube_to_pixel(move_animation_path[start_idx+1].cube)
			set_position(lerp(p1,p2, (move_animation_var-start_idx*(1.0/n))*n))

# convenience
var Stats = m_resource.Stats

func distance(other):
	return tile.cube.distance(other.tile.cube)

## setup

func _ready():
	hex_position = position
	
	if m_resource:
		var template = TemplateConstructor.unit_data[m_resource.unit_class]
		set_texture(load(template.sprite_path))
		
	if m_resource:
		# stat bars
		health_bar.set_max_value(m_resource.stats[Stats.Health])
		health_bar.fill()
		stamina_bar.set_max_value(m_resource.stats[Stats.Stamina])
		stamina_bar.fill()
		morale_bar.set_max_value(m_resource.stats[Stats.Morale])
		morale_bar.fill()
		
	if m_resource:
		# triggers
		for proc in m_resource.proc_group:
			proc.setup()

	if m_team:
		sprite.flip_h = true if m_team.facing == 1 else false
		
	# ui_layer
	if m_resource:
		stacking_hbox.set_value(m_resource.mvpt)
		call_deferred('defer_layout')
#		m_resource.mvpt = 0 # this gets updated on upkeep

func defer_layout():
	stacking_hbox.set_position($UILayer/barbox.get_position()+%HealthBar.get_position())

## get/set

func set_team(team):
	m_team = team
	m_resource.team_name = team.name
	
func get_team():
	return m_team

func set_texture(texture):
	sprite.texture = texture

func get_current_stats():
	return m_resource.stats

func get_stat(x):
	return m_resource.get_stat(x)
	
func set_stat(x, value):
	m_resource.stats[x] = value
	
func modify_bonus_stat(x, value):
	m_resource.bonus_stats[x] += value
	
func get_base_stat(x):
	return m_resource.base_stats[x]
	
func set_base_stat(x, value):
	m_resource.base_stats[x] = value

func get_movement_cost(_tile):
	return m_resource.movecost[_tile.terrain]


## upkeep

func upkeep():
	# recover action
	acted = false
	moved = false
	# recover movement
	recover_move_points(m_resource.get_speed() - m_resource.mvpt)
	# record starting tile
	m_resource.origin_tile = self.tile
	# ...
	m_resource.process_upkeep_triggers()

func downkeep():
	# recover stamina
	if !acted and !moved:
		recover_stamina(2)
	#
	m_resource.process_downkeep_triggers()

## actions

func attack(other):
	# before damage calculation. allow other to process a trigger on being attacked
	# note. I am not a huge fan of adding these triggers just for one skill
	other.m_resource.process_triggers(TokenResource.Proc.PRE_HIT)
	
	# calculate damage (1 minimum damage)
	var damage = 0
	if m_resource.skill.has("Magic Strike"):
		damage = max(get_stat(Stats.Attack) - other.get_stat(Stats.Resistance), 1)
	else:
		damage = max(get_stat(Stats.Attack) - other.get_stat(Stats.Defense), 1)
		
	var stamina_cost = 2 if moved else 1
	expend_stamina(stamina_cost)
	other.take_damage(damage)
	acted = true
	#
	m_resource.process_melee_triggers()
	other.m_resource.process_triggers(TokenResource.Proc.POST_HIT)

func animate_attack(other):
	melee_animation_vector = grid_distance * position.direction_to(other.position)
	animate.play("melee")
	await animate.animation_finished # WARN. possible to interact with token during death animation
	
	
func counter(other):
	# 0 minimum damage
	var damage = max(get_stat(Stats.Counter) - other.get_stat(Stats.Defense), 0)
	expend_stamina(1)
	other.take_damage(damage)
	
func animate_counter(other):
	melee_animation_vector = grid_distance * position.direction_to(other.position)
	animate.play("melee")
	await animate.animation_finished

func attack_or_counter(other):
	if other.m_resource.first_strike and !m_resource.first_strike:
		if other.can_counter(self):
			other.counter(self)
			await other.animate_counter(self)
		if can_attack(other):
			attack(other)
			await animate_attack(other)
	else:
		attack(other)
		await animate_attack(other)
		if other.can_counter(self):
			other.counter(self)
			await other.animate_counter(self)

func ranged_attack(other):
#	other.m_resource.process_triggers(TokenResource.Proc.PRE_HIT)

	# calculate damage (1 minimum damage)
	var damage = 0
	if m_resource.skill.has("Magic Shot"):
		damage = max(get_stat(Stats.RangedAttack) - other.get_stat(Stats.Resistance), 1)
	else:
		damage = max(get_stat(Stats.RangedAttack) - other.get_stat(Stats.RangedDefense), 1)
	var stamina_cost = 2 if moved else 1
	expend_stamina(stamina_cost)
	m_resource.stats[Stats.Ammo] -= 1 # expend ammo
	expend_move_points(1)
	other.take_damage(damage)
	acted = true
	#
#	m_resource.process_melee_triggers()
#	other.m_resource.process_triggers(TokenResource.Proc.POST_HIT)

func animate_ranged_attack(other):
	# push the target back a tiny bit (up to a cap for multiple attacks)
	var push = other.create_tween()
	var push_value = 0.05
	var max_push_value = 0.14 # not an exact multiple of push_value
	var direction = position.direction_to(other.position)
	var displacement = direction * push_value * grid_distance 
	var final_position = other.position + displacement
	var tile_position = hexmap.cube_to_pixel(other.tile.cube)
	if final_position.distance_to(tile_position) >= max_push_value * grid_distance:
		var inter = Geometry2D.segment_intersects_circle(
			other.position, final_position, tile_position, max_push_value * grid_distance)
		if inter != null:
			final_position = other.position + inter * displacement
	push.tween_property(other, "position", final_position, 0.1)
	
	# animate a line
	var init_value = 0.4
	var end_value = 0.2
	var animated_line = ProjectileLine.new(
		position + init_value * grid_distance * direction,
		other.position - end_value * grid_distance * direction
		)
	hexmap.animation_layer.add_child(animated_line)

func take_damage(value):
	var new_health = max(get_stat(Stats.Health) - value, 0)
	set_stat(Stats.Health, new_health)
	health_bar.update(new_health)
	animate_damage_number(value)
	if new_health == 0:
		kill.emit(self)
	m_resource.process_triggers(TokenResource.Proc.TAKE_DAMAGE)
	check_injury()
	stat_changed.emit()
		
func animate_damage_number(value):
	var damage_number =	FloatText.instantiate()
	add_child(damage_number)
	damage_number.show_damage(value)
	await damage_number.animate.animation_finished
	damage_number.queue_free()

func check_injury():
	if m_resource.skill.has("Berserk"):
		return
	var health = m_resource.stats[Stats.Health]
	var base_health = m_resource.base_stats[Stats.Health]
	for stat in [Stats.Attack, Stats.Counter, Stats.RangedAttack]:
		m_resource.health_modifier[stat] = clamp(0.5 + health/base_health, 0, 1)
	m_resource.update_stat_modifier()
	
func expend_stamina(value):
	if m_resource.skill.has("Tireless"):
		return
	var new_stamina = max(get_stat(Stats.Stamina) - value, 0)
	set_stat(Stats.Stamina, new_stamina)
	check_exhaustion()
	stamina_bar.update(new_stamina)
	stat_changed.emit()

func check_exhaustion():
	# stat loss due to low stamina
	var stamina = m_resource.stats[Stats.Stamina]
	for stat in [Stats.Attack, Stats.Counter, Stats.RangedAttack]:
		m_resource.stamina_modifier[stat] = min(0.4 + 0.1 * stamina, 1.0)
	m_resource.stamina_modifier[Stats.Speed] = 0.5 if stamina <= 4 else 1.0
	var defense_modifier = 0.5 if stamina <= 0 else 1.0
	for stat in [Stats.Defense, Stats.RangedDefense, Stats.Resistance]:
		m_resource.stamina_modifier[stat] = defense_modifier
	m_resource.update_stat_modifier()

func recover_stamina(value):
	var new_stamina = min(get_stat(Stats.Stamina) + value, get_base_stat(Stats.Stamina))
	set_stat(Stats.Stamina, new_stamina)
	stamina_bar.update(new_stamina)
	check_exhaustion()
	stat_changed.emit()

func move_path(path, mv_cost):
	# calculate stamina cost
	var cost = 0
	for idx in range(path.size()-1):
		# iterate through tiles and compute the stamina cost
		var _tile = path[idx]
		cost += m_resource.terrain_stamina_cost[_tile.terrain]
	expend_stamina(cost)
	expend_move_points(min(mv_cost, m_resource.mvpt))
	# update tile
	var new_tile = path[0]
	assert(new_tile.is_empty())
	self.tile.clear_slot()
	new_tile.fill_slot(self)
	# 
	moved = true
	m_resource.process_move_triggers()

func recover_move_points(value):
	print('recover move points ', str(self))
	var new_value =  m_resource.mvpt + value
	m_resource.mvpt = new_value
	stacking_hbox.set_value(new_value)
	stat_changed.emit()
	
func expend_move_points(value):
	var new_value = max(m_resource.mvpt - value, 0)
	m_resource.mvpt = new_value
	stacking_hbox.set_value(new_value)
	stat_changed.emit()

# TODO replace hexmap reference with signal
#      
func animate_move_path(path):
	# just animation. also call move_path()
	var p = path.duplicate()
	p.reverse()
	self.move_animation_path = p
	self.move_animation_var = 0
	animate.play("move_path")
	await animate.animation_finished
	hexmap.snap(self, path[0])  

## capabilities

func is_alive():
	return get_stat(Stats.Health) > 0

func capable_move():
	return !acted and get_stat(Stats.Stamina) > 0 and m_resource.mvpt > 0

func can_move(candidate_tile):
	return candidate_tile != self.tile and capable_move()

func _can_melee(other):
	return (
		is_alive() and self != other and self.m_team != other.m_team 
		and get_stat(Stats.Stamina) > 0
		)

func can_attack(other):
	return _can_melee(other) and !self.acted

func can_counter(other):
	return _can_melee(other)
	
func can_shoot(other):
	return (
		!self.acted
		and is_alive() and self != other and self.m_team != other.m_team 
		and get_stat(Stats.RangedAttack) > 0 and get_stat(Stats.Ammo) > 0 
		and self.distance(other) <= get_stat(Stats.Range) 
		)

func capable_ranged_attack():
	return (
		get_stat(Stats.RangedAttack) > 0 and get_stat(Stats.Stamina) > 0 and m_resource.mvpt > 0
		and get_stat(Stats.Ammo)
		)

## AI behaviour

# TODO move this up to hexmap or even higher
# TODO behaviour trees
func evaluate():
	var action_list = [] # return value
	
	# compute pathing and target list
	var mvpt = m_resource.mvpt
	var tiles = pathing.hexmap.get_tile_range(tile, mvpt+1)
	pathing.clear()
	pathing.dijkstra(tile, tiles, m_resource.movecost)
	var tile_list = pathing.viable(self) # note. modifies movecost of adjacent tiles
	
	var target = null
	if capable_ranged_attack():
		# RANGED
		var targets = pathing.viable_ranged_targets(self, tile_list)
		if targets.is_empty():
			return action_list # do nothing
		# just get the first viable target
		target = targets.keys()[0]
		# just get the first viable tile
		var move_target = targets[target][0]
		if move_target != self.tile:
			var path = pathing.path_to(move_target)
			var move_action = MoveAction.new(self, path, pathing.dist[move_target])
			action_list.append(move_action)
		
		var ranged_action = RangedAction.new(self, target)
		action_list.append(ranged_action)
		
	
	if !target:
		# MELEE
		var targets = pathing.viable_targets(self, tile_list)
		# just get the first viable move
		if targets.is_empty():
			# repeat pathing for the whole hexmap
			pathing.clear()
			pathing.dijkstra(tile, pathing.hexmap.tiles.values(), m_resource.movecost)
			# filter for tiles that are adjacent to an attack target 
			var viable = pathing.viable_far(self)
			if viable.is_empty():
				return [] # do nothing
			var far_tile = pathing.min_key(viable)
			var far_path = pathing.path_to(far_tile)
			# step through the path and stop when the distance is greater than speed
			var path = []
			for i in range(far_path.size()):
				var step = far_path[i]
				var move_cost = pathing.dist[step]
				if move_cost <= mvpt or i == far_path.size()-2: # within move distance or adjacent
					path.append(step)
	
			if path[0] != tile:
				var move_action = MoveAction.new(self, path, pathing.dist[path[0]])
				return [move_action]
			else:
				print("warning far pathing algorithm chose current tile")
				return [] # do nothing

		target = targets.keys()[0]
	
		if target.distance(self) > 1:
			# if not adjacent, we need to move
			# Note. targets dictionary values not used, only keys
			var path = pathing.get_cheapest_path(target.tile)
			var move_action = MoveAction.new(self, path, pathing.dist[path[0]])
			action_list.append(move_action)
	
		# melee. check for counters here in the evaluation 
		var melee_action = MeleeAction.new(self, target)
		if target.can_counter(self):
			var counter_action = CounterAction.new(target, self)
			if target.m_resource.first_strike and !m_resource.first_strike:
				action_list.append_array([counter_action, melee_action])
			else:
				action_list.append_array([melee_action, counter_action])
		else:
			action_list.append_array([melee_action])
	
	return action_list
	
##

# TODO plan is to just inspect the skills directory and any skillresource in there gets added to a lookup table

func assemble_triggers():
		# initialise skill indicators
	for key in m_resource.skill.keys():
		m_resource.skill[key].gain(m_resource)

#		match skill_name:
#			"Charge":
#				var proc = load("res://core/combat_model/proc/charge.gd").new(m_resource, skill_value)
#				m_resource.add_trigger(proc)
#			"First Strike":
#				pass
#			"Parry":
#				var proc = load("res://core/combat_model/proc/parry.gd").new(m_resource, skill_value)
#				m_resource.add_trigger(proc)
#			"Magic Shot", "Magic Strike":
#				pass
#			"Berserk", "Tireless", "Intrepid":
#				pass
#			"Forest Expert":
#				m_resource.movecost[HexMap.Tile.Terrain.FOREST] = 1
#				m_resource.terrain_stamina_cost[HexMap.Tile.Terrain.FOREST] = 0
#			"Hills Expert":
#				m_resource.movecost[HexMap.Tile.Terrain.HILL] = 1
#				m_resource.terrain_stamina_cost[HexMap.Tile.Terrain.HILL] = 0
#			"Swamp Expert":
#				m_resource.movecost[HexMap.Tile.Terrain.SWAMP] = 1
#				m_resource.terrain_stamina_cost[HexMap.Tile.Terrain.SWAMP] = 0
#			"Rage":
#				var proc = load("res://core/combat_model/proc/rage.gd").new(self, skill_value)
#				m_resource.add_trigger(proc)
#			_:
#				print("skill not implemented %s" % skill_name)


func _to_string():
	return get_display_name()
