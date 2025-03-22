extends Node

# setup
var hexmap

# path data
var dist = {}
var prev = {}
var unvisited = []

func _ready():
	pass
	
func clear():
	dist.clear()
	prev.clear()
	unvisited.clear()

# https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
# breadth first search 
func dijkstra(start, tiles, cost):
	# use distance = speed + 1 to evaluate options for melee units
	# if start is null 
	
	# Initialize distances
	for tile in tiles:
		# minimum cost of a move is 1 so only consider tiles in this range
		dist[tile] = INF
		prev[tile] = null
		unvisited.append(tile)
	
	if start: # start can be null if we are extending dijkstra's search filed
		dist[start] = 0
	
	while not unvisited.is_empty():
		# find node with smallest distance
		var current = null
		var min_distance = INF
		for tile in unvisited:
			if dist[tile] < min_distance:
				min_distance = dist[tile]
				current = tile
		if current == null:
			break # no reachable tiles
		
		unvisited.erase(current)
		
		# update distances
		for adj_tile in current.adjacent:
			if adj_tile == null:
				continue # boundary
			if not adj_tile in unvisited:
				continue # visited
			var alt = dist[current] + cost[adj_tile.terrain] if adj_tile.is_empty() else INF
			if alt < dist[adj_tile]:
				dist[adj_tile] = alt
				prev[adj_tile] = current

func has(tile):
	return tile in dist
	
func path_to(tile):
	# path is from the target to the start location
	var path = [tile]
	while prev[tile]:
		tile = prev[tile]
		path.append(tile)
	return path
	
func viable(token): 
	# viable movement tiles
	# Note. dist/prev are modified to include adjacent tiles
	
	# get paths which are not occupied and within the speed value
	var tile_lst = [token.tile]
	for tile in dist:
		if not prev[tile]:
			continue # no path from here
		if tile.slot:
			continue # occupied
		if dist[tile] <= token.m_resource.mvpt:
			tile_lst.append(tile) # viable

	# include adjacent tiles if they are not included already && this is the first move this turn
	if !token.moved:
		for tile in token.tile.adjacent:
			if tile and !tile.slot and tile not in tile_lst and token.get_movement_cost(tile) < INF:
				# move to adjacent tile at cost of all movement points
				tile_lst.append(tile)
				dist[tile] = token.m_resource.mvpt
				prev[tile] = token.tile
			
	return tile_lst

func viable_targets(token, viable_tiles):
	var targets = {} # token : [tile]
	for tile in dist.keys(): # looping over all tiles that have been considered by dijkstra
		if tile.slot and token.can_attack(tile.slot):
			for adj_tile in tile.adjacent:
				if adj_tile in viable_tiles:
					update_choices(targets, tile.slot, adj_tile)
	return targets
	
func viable_far(token):
	var far = {} # tile : [distance]
	for tile in dist.keys(): # looping over all tiles that have been considered by dijkstra
		if tile.slot and token.can_attack(tile.slot):
			for adj_tile in tile.adjacent:
				if adj_tile:
					if dist[adj_tile] < far.get(adj_tile, INF):
						far[adj_tile] = dist[adj_tile]
	return far

func viable_ranged_targets(token, viable_tiles):
	var targets = {}
	var rg = token.get_stat(TokenResource.Stats.Range)
	# faster to iterate the enemy positions and check range against all our tiles
	# or we iterate the viable tiles for enemy positions?
	for unit in token.get_team().get_opposed().get_active_unit_list():
		# search for a tile where can hit this unit
		for tile in viable_tiles:
			var target_distance =  unit.tile.cube.distance(tile.cube)
			if target_distance <= rg and target_distance > 1:
				update_choices(targets, unit, tile)
	return targets
				
	
func get_cheapest_path(target_tile):
	# find the cheapest path to the target tile from the available adjacent tiles
	var curr_dist = INF
	var tile = null
	for adj_tile in target_tile.adjacent:
		if adj_tile and self.has(adj_tile):
			var d = dist[adj_tile]
			if d < curr_dist:
				tile = adj_tile
				curr_dist = d
	return path_to(tile) if tile else null

# todo move to utils
func min_key(dct):
	var d = INF
	var min_k = null
	for k in dct.keys():
		if dct[k] < d:
			d = dct[k]
			min_k = k
	return min_k

func update_choices(dct, item, choice):
	if item in dct:
		dct[item].push_back(choice)
	else:
		dct[item] = [choice]
		
