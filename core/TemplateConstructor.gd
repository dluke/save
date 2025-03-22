extends Node

var Token = preload("res://core/token.tscn")


var skills = Dictionary()

@onready var unit_data = process_list_to_dictionary(load_json("res://data/units.json"));

## setup
	
func load_skills(path):
	var dir = DirAccess.open(path)
	for file in dir.get_files():
		var pair = file.split('.')
		var ext = pair[1]
		if ext == 'tres':
			var resource = load(path+file)
			skills[resource.name] = resource


func _ready():
	load_skills('res://core/combat_model/skill/')

## object factory
func new_class_unit(unit_class, unique_name):
	var data = unit_data[unit_class]
	return construct(data, unique_name)
	
func construct(data, unique_name):

	var tkr = TokenResource.new()
	tkr.name = unique_name
	tkr.unit_class = data["class"]
	tkr.base_stats = data["stats"].duplicate()
	tkr.stats = tkr.base_stats.duplicate()
	var skills = data.get("skills", [])

	var token = Token.instantiate()
	token.m_resource = tkr
	token.m_name = unique_name
	
	# initialise skill indicators
	for pair in skills:
		var name = pair[0]
		var value = pair[1]
		var skill = self.skills[name].duplicate()
		skill.value = value
		tkr.skill[name] = skill
	token.assemble_triggers()
	
	return token

## process JSON
func process_item(data):
	data['sprite_path'] = "res://assets/" + data['sprite_path']
	return data
	
func process_list_to_dictionary(lst):
	var dct = Dictionary()
	for item in lst:
		dct[item['class']] = process_item(item)
	return dct

func load_json(path):
	var file = FileAccess.open(path, FileAccess.READ)
	var json = JSON.new()
	var error = json.parse(file.get_as_text())
	if error == OK:
		pass # do nothing
	else:
		print("Error parsing JSON: ", json.error_string())
	return json.data

