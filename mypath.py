'''
	Author:	Sungguk Cha
	eMail :	navinad@naver.com
'''

class Path(object):
	@staticmethod
	def db_root_dir(dataset):
		if dataset == 'caltech101':
			return './caltech101/101_ObjectCategories'
		else:
			print("Dataset {} is not available.".format(dataset))
			raise NotImplementedError
