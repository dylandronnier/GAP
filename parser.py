import numpy as np

def loadDatabase(databaseName, rescaleFunction=None, first=None):
	print 'Loading %s database' % (databaseName)
	all_config = [a for a in parseDatabase(databaseName)]
	head = first if first else len(all_config)
	
	print ' Total configurations = %d' % len(all_config)
	print ' Getting first %d' % head 

	data = [ a for a in all_config[0:head]]
	
	if rescaleFunction:
		print '   rescaling it '
		return [ rescaleFunction(a) for a in data]
	return data

def parseLine(line):
	tokens = line.split()

	neighbours = int(tokens[0])
	
	x, y, z, e, Fx, Fy, Fz = [ float(t) for t in tokens[1:8]]

	data = [float(t) for t in tokens[8:]]

	assert (len(data) % 3) == 0
	
	n_data = np.array( [np.array(data[i: i+3]) for i in xrange(0, len(data), 3)])
	
	atom_description =  {
		'coord'   : np.array([x, y, z]),
		'e'       : e,
		'F'       : np.array([Fx, Fy, Fz]),
		'n_count' : neighbours,
		'n_data'  : n_data 
	}

	return atom_description

def parseDatabase(filename):
	for l in open(filename, 'r'):
		l = l.strip()
		if not l or l.startswith('#'):
			continue

		yield parseLine(l)

