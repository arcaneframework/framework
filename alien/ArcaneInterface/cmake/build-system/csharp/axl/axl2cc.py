#! @PYTHON_EXECUTABLE@

import arcanelaunch
from arcanelaunch import setenv,getenv
import sys
import os
import copy
import shutil
import optparse
import re

link_dirs = "@ARCANE_LINK_DIRECTORIES@"
#TODO: traiter correctement les espaces dans les chemins
link_dirs.replace(" ",os.pathsep)
#print "link_dirs=",link_dirs

path_bin = "@ARCANE_INSTALL_BIN@"
path_lib = "@ARCANE_INSTALL_LIB@"
path_shr = "@ARCANE_INSTALL_SHR@"

stdenv_exe = os.path.join(path_lib,"arcane_axl") + getenv("STDENV_PURE","")
setenv("STDENV_PARALLEL","FALSE")
setenv("STDENV_APPLICATION_NAME","axl2cc")
setenv("STDENV_QUIET","TRUE")
setenv("STDENV_TRACE","off")

nb_arg = len(sys.argv)
pargs = []
do_copy = False
if nb_arg==6 and sys.argv[5]==".xml":
	# Vieux format.
	print "WARNING: this format is deprecated. Use axl"
	path = sys.argv[2]
	component_name = sys.argv[3]
	name = sys.argv[4]
	extension = sys.argv[5]
	pargs = copy.copy(sys.argv)
	full_name = os.path.join(path,name) + extension
else:
	parser = optparse.OptionParser(usage="%prog [-i header] [-o output_path] axlfile")
	#print "nb_arg",nb_arg
	parser.add_option("-i","--header-path",type="string",dest="header_path",help="header sub path")
	parser.add_option("-o","--output-path",type="string",dest="output_path",help="path to write output files")
	parser.add_option("-c","--copy",action="store_true",dest="do_copy",help="true if installing in share path")
	(options, args) = parser.parse_args()
	print str(options)
	print str(args)
	output_path = os.getcwd()
	if options.output_path:
		output_path = options.output_path
	print "OutputPath=",output_path
	component_name = "."
	if options.header_path:
		component_name = options.header_path
	if len(args)!=1:
		parser.error("axl file not specified")
		sys.exit(1)
	full_name = args[0]								
	file_name = os.path.basename(full_name)
	file_path = os.path.dirname(full_name)
	if len(file_path)==0:
		file_path = "."
	file_name_no_extension_re = re.compile("(.*)\.axl").match(file_name)
	if file_name_no_extension_re == None:
		parser.error("axlfile has to have extension '.axl'")
		sys.exit(1)
	file_name_no_extension = file_name_no_extension_re.group(1)
	print "Infos: file_path=",file_path," name=",file_name_no_extension
	name = file_name_no_extension
	extension = ".axl"
	pargs.append(sys.argv[0])
	pargs.append(output_path)
	pargs.append(file_path)
	pargs.append(component_name)
	pargs.append(file_name_no_extension)
	pargs.append(".axl")
	
output_name =  os.path.join(path_shr,name)
if component_name != ".":
	output_name += "_" + component_name
output_name += extension

al = arcanelaunch.ArcaneLaunchExec()
al.setApplicationExecutable(stdenv_exe)
al.setParallelService(None)
al.addToLdPath(link_dirs)
r = al.process(pargs)
if r == 0 and do_copy:
	print "Installing file input=",full_name,"output=",output_name
	shutil.copy(full_name,output_name)
print "Return value: v=",r
sys.exit(r)
