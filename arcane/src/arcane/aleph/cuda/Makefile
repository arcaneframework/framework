TMP= $(shell cea_home -t -st)
HOSTNAME=$(shell cea_machine)
ARCANE_SRC=$(TMP)/arcane/src

what:;@echo $(HOSTNAME)

cuda:
	make -f Makefile.cu

#MAKE_ALL = -B

$(HOSTNAME):
	ssh camierjs@$(HOSTNAME) '(\
cd arcane/obj/$(HOSTNAME)/src/arcane/aleph && \
make $(MAKE_ALL))'

cartan:HOSTNAME=cartan
cartan:$(HOSTNAME)



################
# /TMP COMPILE #
################
SRCs=AlephKernel AlephMatrix AlephService AlephTopology AlephVector

CPP = /usr/bin/g++
CPPFLAGS	= -c -Wall -O2
TMP_INCLUDE=-I/tmp/arcane/src/arcane/src/ -I/tmp/arcane/src/arcane/src/arcane -I/tmp/arcane/obj/x-airen
Os = $(SRCs:%=%.o)

tmp:$(Os)

%.o: %.cc
	$(CC) $(CPPFLAGS) $(TMP_INCLUDE) -o $@ $*.cc

#########
# Clean #
#########
cln:
	@echo "Cleaning..."
	@/bin/rm -rf *~			# doublons XEmacs
	@/bin/rm -rf ~*			# doublons NEdit
	@/bin/rm -rf *% 		# doublons texEdit
	/bin/rm -rf Aleph*.o	# fichiers objets
	@/bin/rm -rf core		# fichiers core
	@-rm -f *.o *.a *.so *.co

##########
# BACKUP #
##########
DATE	= $(shell date '+%y%m%d')
HOUR	= $(shell date '+%H%M')
PWD	= $(notdir $(shell pwd))
BKP_NM	= $(PWD).$(DATE).$(HOUR)
bkp:*.cc *.h
	@(cd ../; echo "Taring  '$(BKP_NM)'..."; \
		tar czf $(BKP_NM).tgz $(PWD); \
		mv $(BKP_NM).tgz ../bkp/src/aleph)

