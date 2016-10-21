import argparse
import re
import os
import matplotlib.pyplot as plt
from faster_rcnn.pattern import rpnloss,cnnloss,pvaloss

data = {}
def check_block(pattern,block):
    #get data key
    block_key = re.search(pattern.key_pattern[0],block[0])
    if not block_key:
	raise ValueError('no {} found in {}'.format(desc['key'],block[0]))
    block_key = int(block_key.group(pattern.key_pattern[1]))
    print "key:%d"%(block_key)
    #iteration lines to get data
    data_patterns,data_group_index,_ = pattern.data_pattern
    block_data = []
    for i in range(len(data_patterns)):  
	for line in block:
	    rst = re.search(data_patterns[i],line)
	    if rst:
		block_data.append(rst.group(data_group_index[i]))
	if len(block_data) <= i:
	    raise ValueError("pattern {} didn't find corresponding data in {}".format(_pattern,block))
    data[block_key] = block_data

def parse_log(pattern,logfile_path):
    try:
	with open(logfile_path) as log:
	    lines = log.readlines()
    except IOError as err:
	print "open file {} error!".format(logfile_path)
	exit(-1)
    i = 0
    while i < len(lines):
	check_block(pattern,lines[i:i+pattern.block_size])
	i += pattern.block_size

def plot(pattern,export_image=False,output_path=None):
    result = {}
    data_desc = pattern.data_pattern[2]
    result['key']=[]
    for name in data_desc:
	result[name] = []
    items = data.items()
    items.sort()
    for key,data_block in items:
	result['key'].append(key)
	i = 0
        for name in data_desc:
	    result[name].append(data_block[i])
	    i += 1
    
    #plot
    for name in data_desc:
        plt.plot(result['key'],result[name],label=name)
        plt.xlabel(pattern.key_pattern[2])
        plt.ylabel(name)
        plt.legend()
        #plt.show()
	if export_image:
	    img_path = os.path.join(output_path,name)
	    plt.savefig(img_path+".png")
	    #clear current figure
	    plt.clf()
	else:
	    plt.show()
    #plot all
    for name in data_desc:
	plt.plot(result['key'],result[name],label=name)
    plt.legend()
    if export_image:
	img_path = os.path.join(output_path,"all.png")
	plt.savefig(img_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path',help='log file path',required=True)
    parser.add_argument('--model',dest='parsing_model',action='store',default='cnn',help='prsing what kind of network',choices=['cnn','rpn','pva'],required=True)
    parser.add_argument('--export_image',dest='export',action='store_true')
    parser.add_argument('--output_path',dest='output_path',action='store',default='./',help='path of output images')
    args = parser.parse_args()
    if args.parsing_model == 'rpn':
	pattern = rpnloss
	print 'parsing rpn log......'
    elif args.parsing_model == 'cnn':
	pattern = cnnloss
	print 'parsing cnn log......'
    elif args.parsing_model == 'pva':
	pattern = pvaloss
	print 'parsing pva log......'
    else:
	raise ValueError("invalid model:{}".format(args.parsing_model))
    parse_log(pattern,args.log_path)
    #check if output path valid
    if args.export:
	if not os.path.exists(args.output_path):
	    print '{} not exist,creating......'.format(args.output_path)
	    os.makedirs(args.output_path)
    plot(pattern,args.export,args.output_path)

if __name__=='__main__':
    main()
