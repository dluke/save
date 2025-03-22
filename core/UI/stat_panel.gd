extends Control

@onready var cp_list = [
	%health, %stamina, %morale,
	%attack, %counter, %speed,
	%defense, %ranged_defense, %resistance,
	%ranged_attack, %range, %ammo
	]

func update(token):
	for i in range(len(cp_list)):
		cp_list[i].set_value(token.get_stat(i))
	
