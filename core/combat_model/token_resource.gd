extends Resource
class_name TokenResource


# package up the data and functions for the combat model here
# this separates them from token.gd which is primarily concerned with the View

@export var name : String
@export var team_name : String
@export var tile : Tile
#@export var axial_coord : Vector2 # todo
@export var unit_class : String
@export var base_stats = Array()
@export var stats = Array()
@export var skill = Dictionary()
#@export var effects = Dictionary() # todo

@export var mvpt = 0 # movement points

#
var origin_tile = null

# flags
var first_strike: bool:
	get: return skill.has("First Strike")

# + modifiers
var bonus_stats = Array()
# % modifiers
var health_modifier = Array()
var stamina_modifier = Array()
var morale_modifier = Array()
var stat_modifier = Array() # TODO how to update this? how to combine multiple modifiers?

# plains, forest, hills, swamp, impassable
var movecost = [1, 2, 3, 3, INF] 
var terrain_stamina_cost = [0, 1, 1, 1, INF]

# define stats
enum Stats {
	Health, Stamina, Morale, 
	Attack, Counter, Speed, 
	Defense, RangedDefense, Resistance, 
	RangedAttack, Range, Ammo
	}

func _init():
	bonus_stats.resize(12)
	bonus_stats.fill(0)
	for arr in [health_modifier, stamina_modifier, morale_modifier, stat_modifier]:
		arr.resize(12)
		arr.fill(1)
		
func update_stat_modifier():
	# stat modifier is updated from the modifiers for each primary stat additively
	for i in range(stat_modifier.size()):
		var total = (1-health_modifier[i]) + (1-stamina_modifier[i]) + (1-morale_modifier[i])
		stat_modifier[i] = 1-clamp(total, 0, 1)

func get_base_stat(x):
	return base_stats[x]
	
func get_stat(x):
	return int(ceil(stat_modifier[x] * (stats[x] + bonus_stats[x])))

# convenience functions
func get_health(): return get_stat(Stats.Health)
func get_stamina(): return get_stat(Stats.Stamina)
func get_morale(): return get_stat(Stats.Morale)
#.
#.
func get_speed(): return get_stat(Stats.Speed)


func modify_bonus_stat(x, value):
	bonus_stats[x] += value


# TODO how to simplify this?
enum Proc {UPKEEP, DOWNKEEP, MOVE, MELEE, PRE_HIT, POST_HIT, TAKE_DAMAGE}

# define triggers
var upkeep_proc = [] 
var downkeep_proc = [] 
var move_proc = [] # after move
var melee_proc = [] # after attack
var melee_pre_hit = []
var melee_post_hit = []
var take_damage = []

var proc_lists = [
	upkeep_proc, 
	downkeep_proc, 
	move_proc, 
	melee_proc, 
	melee_pre_hit, 
	melee_post_hit,
	take_damage
]

var proc_group = [] # for calling .setup()

func add_trigger(proc):
	proc_group.append(proc)
	for x in proc.enum_list:
		proc_lists[x].append(proc)

func process_triggers(x: int):
	for item in proc_lists[x]:
		item.forward(x)
		item.play(x)

func process_upkeep_triggers():
	process_triggers(Proc.UPKEEP)

func process_downkeep_triggers():
	process_triggers(Proc.DOWNKEEP)

func process_move_triggers():
	process_triggers(Proc.MOVE)

func process_melee_triggers():
	process_triggers(Proc.MELEE)

# ...

