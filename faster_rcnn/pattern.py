#loss patterns for each supported network

class NetworkPattern(object):
    def __init__(self,name,loss_pattern_dict,loss_block_size):
	assert isinstance(loss_pattern_dict,dict),"loss_pattern_dict must be a dict"
	assert loss_block_size>=1,"invalaid loss_block lines!"
	#model name
	self.name = name
	#the discription of this pattern
	self.desc = loss_pattern_dict['desc']
	self.pattern = loss_pattern_dict
	#line number of a block
	self.loss_block_size = loss_block_size
    
    @property
    def block_size(self):
	return self.loss_block_size
    
    @property
    def key_pattern(self):
	return self.pattern['key'],self.pattern['key_group'],self.pattern['key_desc']
    
    @property
    def data_pattern(self):
	return self.pattern['data'],self.pattern['data_group'],self.pattern['data_desc']
    
    def data_pattern_at(self,index):
	return self.pattern['data'][index],self.pattern['data_group'][index],self.pattern['data_desc'][index]

    
__rpn_patterns={
	'desc':'pattern of faster-rcnn rpn loss',
	#data key
	'key':'Iteration ((\d)+)',
	'key_group':1,
	'key_desc':'Iteration',
	#data values
	'data':[' loss = ((\d)+(\.(\d)+)*)',\
	'(rpn_)?loss_bbox = ((\d)+(\.(\d)+)*)',\
	'(rpn_)?cls_loss = ((\d)+(\.(\d)+)*)',\
	'lr = ((\d)+(\.(\d)+)*)'],
	'data_group':[1,2,2,1],
	'data_desc':['sum_loss',\
	'bbox_loss',\
	'cls_loss',\
	'learning rate']
	}
__cnn_patterns={
	'desc':'pattern of faster-rcnn cnn loss',
	#data key
	'key':'Iteration ((\d)+)',
	'key_desc':'Iteration',
	'key_group':1,
	#data values
	'data':[' loss = ((\d)+(\.(\d)+)*)',\
	' loss_bbox = ((\d)+(\.(\d)+)*)',\
	' loss_cls = ((\d)+(\.(\d)+)*)',\
	'lr = ((\d)+(\.(\d)+)*)'],
	'data_group':[1,1,1,1],
	'data_desc':['sum_loss',\
	'bbox_loss',\
	'cls_loss',\
	'learning rate']
	}
__pva_patterns={
	'desc':'pattern of pva-faster-rcnn loss,which trained train end2end',
	#data key
	'key':'Iteration ((\d)+)',
	'key_group':1,
	'key_desc':'Iteration',
	#data values
	'data':[' loss = ((\d)+(\.(\d)+)*)',\
	' loss_bbox = ((\d)+(\.(\d)+)*)',\
	' cls_loss = ((\d)+(\.(\d)+)*)',\
	' rpn_loss_bbox = ((\d)+(\.(\d)+)*)',\
	' rpn_cls_loss = ((\d)+(\.(\d)+)*)',\
	'lr = ((\d)+(\.(\d)+)*)'],
	'data_group':[1,1,1,1,1,1],
	'data_desc':['sum_loss',\
	'bbox_loss',\
	'cls_loss',\
	'rpn_bbox_loss',\
	'rpn_cls_loss',\
	'learning rate']
	}

rpnloss = NetworkPattern('faster r-cnn rpn loss pattern',__rpn_patterns,4)
cnnloss = NetworkPattern('faster r-cnn cnn loss pattern',__cnn_patterns,4)
pvaloss = NetworkPattern('pva-faster r-cnn loss pattern',__pva_patterns,6)
