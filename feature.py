import sys
sys.path.insert(0,'./lshash/')
from lshash import LSHash

bit_num = 5#18
compare_kernel_num = 10#10

class feature_comparer():
	def __init__(self, fea_dim, compare_thresh):
		self.lsh = LSHash(bit_num,fea_dim,compare_kernel_num)
		self.fv_dict = {}
		self.compare_thresh = compare_thresh

	def load(self, filename):
		f = open(filename, 'r')
		while(1):
			line = f.readline()
			if not line:
				break

			fv = line.split(':')[0]
			id = line.split(':')[1]
			self.fv_dict[fv] = id

			fv_array = []
			s = fv[1:-1].split(',')
			for i in range(0, len(s)):
				fv_array.append(float(s[i]))
			self.lsh.index(fv_array)

	def insert(self, feature, id):
		self.fv_dict[str(feature)[1:-1]] = str(id)
		self.lsh.index(feature)

	def match(self, feature):
		q = self.lsh.query(feature,distance_func='cosine')
		if len(q) == 0:
			return False, -1
		mindis = q[0][1]
		if mindis < self.compare_thresh:
			return True, self.fv_dict[str(q[0][0])[1:-1]]
		else:
			return False, -1
