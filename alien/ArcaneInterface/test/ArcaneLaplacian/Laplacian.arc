<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="Laplacian" xml:lang="en">
	<arcane>
		<title>Exemple Arcane d'un module Hydro très, très simplifié</title>
		<timeloop>LaplacianLoop</timeloop>
	</arcane>

	<arcane-post-processing>
		<output-period>10</output-period>
		<output>
			<variable>CellMass</variable>
			<variable>Pressure</variable>
			<variable>Density</variable>
			<variable>Velocity</variable>
			<variable>NodeMass</variable>
			<variable>InternalEnergy</variable>
			<!-- <variable>CellVolume</variable> -->
		</output>
    <format>
     <binary-file>false</binary-file>
    </format>
	</arcane-post-processing>

	<mesh>
        <file internal-partition='true'>sod.vtk</file>
		<initialisation>
			<variable nom="Density" valeur="1." groupe="ZG" />
			<variable nom="Pressure" valeur="1." groupe="ZG" />
			<variable nom="AdiabaticCst" valeur="1.4" groupe="ZG" />
			<variable nom="Density" valeur="0.125" groupe="ZD" />
			<variable nom="Pressure" valeur="0.1" groupe="ZD" />
			<variable nom="AdiabaticCst" valeur="1.4" groupe="ZD" />
		</initialisation>
	</mesh>

	<module-main></module-main>

	<!-- Configuration du module hydrodynamique -->
	<laplacian>
		<deltat-init>0.001</deltat-init>
		<deltat-min>0.00001</deltat-min>
		<deltat-max>0.0001</deltat-max>
		<final-time>0.05</final-time>

		<boundary-condition>
			<surface>XMIN</surface>
			<type>Vx</type>
			<value>0.</value>
		</boundary-condition>
		<boundary-condition>
			<surface>XMAX</surface>
			<type>Vx</type>
			<value>0.</value>
		</boundary-condition>
		<boundary-condition>
			<surface>YMIN</surface>
			<type>Vy</type>
			<value>0.</value>
		</boundary-condition>
		<boundary-condition>
			<surface>YMAX</surface>
			<type>Vy</type>
			<value>0.</value>
		</boundary-condition>
		<boundary-condition>
			<surface>ZMIN</surface>
			<type>Vz</type>
			<value>0.</value>
		</boundary-condition>
		<boundary-condition>
			<surface>ZMAX</surface>
			<type>Vz</type>
			<value>0.</value>
		</boundary-condition>
		
	</laplacian>
</case>
