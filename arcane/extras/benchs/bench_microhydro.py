#!/usr/bin/env python3
# Note: il faut au moins python 3.5 (pour subprocess.run())

import argparse
import subprocess
from string import Template
from argparse import RawDescriptionHelpFormatter

xstr = """<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Tube a choc de Sod</title>
    <timeloop>ArcaneHydroLoop</timeloop>
    <modules>
      <module name="ArcaneLoadBalance" active="$do_load_balance" />
    </modules>
  </arcane>

  <meshes>
    <mesh>
      <ghost-layer-builder-version>4</ghost-layer-builder-version>
      <generator name="Cartesian3D" >
        <nb-part-x>$nb_part_x</nb-part-x> 
        <nb-part-y>$nb_part_y</nb-part-y>
        <nb-part-z>$nb_part_z</nb-part-z>
        <origin>1.0 2.0 3.0</origin>
        <generate-sod-groups>true</generate-sod-groups>
        <x><n>$nb_cell_x</n><length>2.0</length></x>
        <y><n>$nb_cell_y</n><length>2.0</length></y>
        <z><n>$nb_cell_z</n><length>4.0</length></z>
      </generator>
      <initialization>
        <variable><name>Density</name><value>1.0</value><group>ZG</group></variable>
        <variable><name>Density</name><value>0.125</value><group>ZD</group></variable>

        <variable><name>Pressure</name><value>1.0</value><group>ZG</group></variable>
        <variable><name>Pressure</name><value>0.1</value><group>ZD</group></variable>

        <variable><name>AdiabaticCst</name><value>1.4</value><group>ZG</group></variable>
        <variable><name>AdiabaticCst</name><value>1.4</value><group>ZD</group></variable>
      </initialization>
    </mesh>
  </meshes>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <arcane-load-balance>
    <active>true</active>
    <partitioner name="Metis" />
    <period>5</period>
    <statistics>true</statistics>
    <max-imbalance>0.01</max-imbalance>
    <min-cpu-time>0</min-cpu-time>
  </arcane-load-balance>

  <!-- Configuration du module hydrodynamique -->
  <simple-hydro>
    <deltat-init>0.00001</deltat-init>
    <deltat-min>0.000001</deltat-min>
    <deltat-max>0.0001</deltat-max>
    <final-time>0.2</final-time>

    <viscosity>cell</viscosity>
    <viscosity-linear-coef>.5</viscosity-linear-coef>
    <viscosity-quadratic-coef>.6</viscosity-quadratic-coef>

    <boundary-condition>
      <surface>XMIN</surface><type>Vx</type><value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>XMAX</surface><type>Vx</type><value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>YMIN</surface><type>Vy</type><value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>YMAX</surface><type>Vy</type><value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>ZMIN</surface><type>Vz</type><value>0.</value>
    </boundary-condition>
    <boundary-condition>
      <surface>ZMAX</surface><type>Vz</type><value>0.</value>
    </boundary-condition>
  </simple-hydro>

</case>
"""

epilog_doc = """
Ce script permet de spécifier et d'exécuter le test MicroHydro en MPI
sur N processeurs, chaque processeur ayant le même nombre de mailles.

Il permet donc des tests d'extensibilité faible (weak scaling).

Ce test doit s'exécuter dans le répertoire où Arcane a été compilé.

L'option '-n|--nb-proc' spécifie le nombre de processus. Si ce nombre
est supérieur à 32, il doit être un multiple de 32.

L'option '-s|--mesh-size' indique le nombre de chunks de mailles pour
chaque PE. La taille d'un chunk est de 2000 mailles. Par defaut le
nombre de chunk est de 10.

Il est possible de spécifier un repartitionnement via l'option
'-l|--loadbalance'. Dans ce cas, le repartitionnement aura lieu toutes
les 5 itérations. Arcane doit avec été compilé avec 'ParMetis' pour
que cela fonctionne.
"""

parser = argparse.ArgumentParser(description="MicroHydro bench",formatter_class=RawDescriptionHelpFormatter,epilog=epilog_doc)
required_arguments = parser.add_argument_group('required named arguments')
required_arguments.add_argument("-n","--nb-proc", dest="nb_proc", action="store", help="number of processus", type=int, required=True)
parser.add_argument("-s","--mesh-size", dest="mesh_size", action="store", help="size of mesh", type=int, default=10)
parser.add_argument("-l","--load-balance", dest="do_load_balance", action="store_true", help="true if load balance is activated")
parser.add_argument("-m","--max-iteration", dest="max_iteration", action="store", help="number of iteration to do", type=int, default=100)
parser.add_argument("-p","--arcane-driver-path", dest="arcane_driver_path", action="store", help="arcane_test_driver path", type=str, default="./bin/arcane_test_driver")

args = parser.parse_args()
nb_proc = args.nb_proc
nb_cell_mult = args.mesh_size
if nb_proc>32:
    if nb_proc % 32 != 0:
        raise RuntimeError("Bad number of proc (should be a multiple of 32)")

s = Template(xstr)

# Nombre de parties en (X,Y,Z). X*Y*Z doit etre egal au nombre de PE
nb_part_x = 8
nb_part_y = 4
nb_part_z = (nb_proc // 32)
# En dessous de 32 PE, on découpe de manière spécifique
if nb_proc==24:
    nb_part_x, nb_part_y, nb_part_z = 4, 3, 2
elif nb_proc==16:
    nb_part_x, nb_part_y, nb_part_z = 4, 2, 2
elif nb_proc==12:
    nb_part_x, nb_part_y, nb_part_z = 3, 2, 2
elif nb_proc==8:
    nb_part_x, nb_part_y, nb_part_z = 2, 2, 2
elif nb_proc==4:
    nb_part_x, nb_part_y, nb_part_z = 2, 2, 1
elif nb_part_z==0:
    nb_part_x, nb_part_y, nb_part_z = nb_proc, 1, 1

total_nb_part = nb_part_x * nb_part_y * nb_part_z

# Nombre de mailles en (X,Y,Z)
nb_cell_x = 20 * nb_part_x
nb_cell_y = 20 * nb_part_y
nb_cell_z = 5  * nb_part_z * nb_cell_mult
total_nb_cell = nb_cell_x * nb_cell_y * nb_cell_z
do_load_balance = "true" if args.do_load_balance else "false"
d = {
    "nb_part_x" : nb_part_x, "nb_part_y" : nb_part_y, "nb_part_z" : nb_part_z,
    "nb_cell_x" : nb_cell_x, "nb_cell_y" : nb_cell_y, "nb_cell_z" : nb_cell_z,
    "do_load_balance" : do_load_balance
    }

z = s.substitute(d)
print(z)

print("TotalNbCell=",total_nb_cell," (per part=",total_nb_cell//total_nb_part,")")
case_file = open("test.arc",mode="w")
case_file.write(z)
case_file.close()

command = [ args.arcane_driver_path, "launch", "-n", str(nb_proc), "-m", str(args.max_iteration), "-We,ARCANE_NEW_MESHINIT,1", "test.arc" ]
print(command)
subprocess.run(command)
