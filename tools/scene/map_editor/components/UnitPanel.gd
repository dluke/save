extends PanelContainer

# click and drag  or click select?

signal pickup_unit 

var UnitSquare = preload("res://tools/scene/map_editor/components/unit_square.tscn")

@onready var grid = $vbox/ALL/GridContainer

func _ready():
	var unit_data = TemplateConstructor.unit_data
	for key in unit_data:
		var data = unit_data[key]
		var unit_square = UnitSquare.instantiate()
		grid.add_child(unit_square)
		unit_square.set_data(data)
		connect_square(unit_square)

func connect_square(unit_square):
	unit_square.pickup_unit.connect(_on_pickup_unit)

func _on_pickup_unit(data):
	pickup_unit.emit(data) # propagate up
	
