import numpy as np

class FormatCsv():
	"""
	This class formats prediction.csv to submission.csv to follow the
	kaggle submission guidelines
	"""

	def formatcsv(self, t = "test.csv", p = "prediction.csv", s = "submission.csv"):
		"""
		params:
		t = path to test.csv -> query images (from kaggle)
		p = path to prediction.csv -> prediction made by the model
		s = path to output file that will be created
		"""
		temp = np.loadtxt(open(t, "rb"), delimiter=",", skiprows=1, dtype="str")
		tids = temp[...,0]		# test ids

		temp = np.loadtxt(open(p, "rb"), delimiter=",", skiprows=1, dtype="str")
		d = {i:j for i,j in temp}	# dict of ids, predictions in predictions.csv


		op = open(s, "w")	# this file will be submitted to kaggle
		op.write("id,url\n")

		for i in tids:
			_id = i[1:-1]	# excluding quotes
			if _id in d:
				op.write(_id+","+d[_id]+"\n")
			else:
				op.write(_id+","+"\n")

		op.close()
