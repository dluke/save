extends PanelContainer

var SkillBoxItem = preload("res://core/UI/tactics_card/skillbox_item.tscn")

# components
@onready var skillbox = %skillbox
@onready var statsbox = %statsbox
@onready var texturebox = %texturebox
@onready var health_bar = %HealthBar
@onready var stamina_bar = %StaminaBar
@onready var morale_bar = %MoraleBar

@onready var cp_list = [
	%attack, %counter, %speed,
	%defense, %ranged_defense, %resistance,
	%ranged_attack, %range, %ammo
	]

var token 

func connect_token(_token):
	token = _token
	token.stat_changed.connect(self.update)
	var template = TemplateConstructor.unit_data[token.m_resource.unit_class]
	texturebox.set_texture(load(template.sprite_path))
	# 
	health_bar.set_max_value(token.m_resource.get_health())
	health_bar.fill()
	stamina_bar.set_max_value(token.m_resource.get_stamina())
	stamina_bar.fill()
	morale_bar.set_max_value(token.m_resource.get_morale())
	morale_bar.fill()
	#
	for i in range(9):
		cp_list[i].set_value(token.get_stat(i+3))
	# skills/effects
	for key in token.m_resource.skill.keys():
		var skill = token.m_resource.skill[key]
		var item = SkillBoxItem.instantiate()
		# TODO add scripts to skillbox and skillboxitem to handle setup gracefully
		skillbox.get_node('vbox').add_child(item)
		item.get_node('texture').texture = skill.texture
		item.get_node('label').text = skill.name
	

func update():
	#
	health_bar.update(token.m_resource.get_health())
	stamina_bar.update(token.m_resource.get_stamina())
	morale_bar.update(token.m_resource.get_morale())
	#
	for i in range(9):
		cp_list[i].set_value(token.get_stat(i+3))
	
	
