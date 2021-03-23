La configuration du code {#arcanedoc_codeconfig}
=======================

# Introduction {#arcanedoc_codeconfig_intro}

La configuration du code est décrite dans un fichier externe,
dont le nom est CODE.config, avec \a CODE le nom du code.
  
Ce fichier décrit l'ensemble des boucles en temps
(voir \ref arcanedoc_timeloop) disponibles pour
le code, ainsi que leur configuration.

# Structure du fichier {#arcanedoc_codeconfig_struct}

Ce fichier de configuration de l'application est au format XML.
Voici un exemple d'un tel fichier pour un module <em>MicroHydro</em> qui
se trouve dans le répertoire des exemples de ARCANE (\c samples):

~~~~~~~~~~~~~~~~~~~~~{.xml}
<?xml version="1.0" encoding="UTF-8"?>
<arcane-config code-name="MicroHydro">
 <time-loops>
  <time-loop name="MicroHydroLoop">
   <title>MicroHydro</title>
   <description>Boucle en temps de l'exemple Arcane MicroHydro</description>

   <modules>
    <module name="MicroHydro" need="required" />
    <module name="ArcanePostProcessing" need="required" />
   </modules>

   <singleton-services>
   </singleton-services>

   <entry-points where="init">
    <entry-point name="MicroHydro.HydroStartInit" />
   </entry-points>

   <entry-points where="compute-loop">
    <entry-point name="MicroHydro.ComputePressureForce" />
    <entry-point name="MicroHydro.ComputeVelocity" />
    <entry-point name="MicroHydro.ApplyBoundaryCondition" />
    <entry-point name="MicroHydro.MoveNodes" />
    <entry-point name="MicroHydro.ComputeGeometricValues" />
    <entry-point name="MicroHydro.UpdateDensity" />
    <entry-point name="MicroHydro.ApplyEquationOfState" />
    <entry-point name="MicroHydro.ComputeDeltaT" />
   </entry-points>

  </time-loop>
 </time-loops>
</arcane-config>
~~~~~~~~~~~~~~~~~~~~~

## L'élément <time-loops>

L'ensemble des boucles en temps est décrit dans l'élément
`<time-loops>. Chaque boucle en temps est représentée
par l'élément `<time-loop>` et identifiée par son nom (attribut
`name`). Le fichier précédent décrit donc une seule boucle en temps 
nommée <em>MicroHydroLoop</em>.
  
Outre le titre et la description de la boucle, on remarque 3 éléments :
`<modules>`, `<singleton-services>` et `<entry-points>`.

### L'élément <modules>

Cet élément décrit l'ensemble des modules du code nécessaire à
l'exécution de la boucle en temps. L'attribut `name` identifie
le module par son nom et l'attribut `need` (valant **required**
ou **optional**) indique si le module doit obligatoirement être
présent ou non. Si le module n'est pas obligatoire et qu'il
n'est pas fourni à l'exécution (absence de la bibliothèque
du module), ses points d'entrée seront ignorés. Cela permet
de construire des variantes d'une même boucle en temps.

### L'élément <singleton-services>

Cet élément est assez similaire à `<modules>` et décrit
l'ensemble des services singletons du code utilisés lors
de l'exécution. Un service singleton est un service pour lequel il
n'existe qu'une seule instance qui est créée lors de
l'initialisation du code. Un tel service peut avoir des options dans
le jeu de données. La spécification des services singletons se fait
comme suit:

~~~~~~~~~~~~~~~~~~~~~{.xml}
<singleton-services>
 <service name="Toto" need="required">
 <service name="Tutu" need="optional">
</singleton-services>
~~~~~~~~~~~~~~~~~~~~~
  
Comme pour le module, il y a deux attributs `name`
et `need`. Si le service est optionel et qu'il n'est pas trouvé,
il ne sera pas instantié. Dans le code, il est possible de
récupérer un service singleton dont on connait l'interface, via
la classe ServiceBuilder.

### L'élément <entry-points>

Cet élément contient l'attribut `where` précisant
l'endroit d'appel des différents points d'entrée. Les valeurs possibles
sont décrites dans le tableau ci-dessous:

<table>

<tr>
<th>Valeur</th>
<th>Description</th>
</tr>

<tr>
<td> **build** </td>
<td>Appel du point d'entrée à la création du module. Lors
de la création du module, le maillage n'est pas encore chargé et il
ne doit donc pas être utilisé.</td>
</tr>

<tr>
<td> **init** </td>
<td>Appel du point d'entrée lors de l'initialisation du code.</td>
</tr>

<tr>
<td> **compute-loop** </td>
<td>Appel du point d'entrée dans la boucle des itérations.</td>
</tr>

<tr>
<td> **restore** </td>
<td>Appel du point d'entrée lors d'un retour arrière.</td>
</tr>

<tr>
<td> **on-mesh-changed** </td>
<td>Appel du point d'entrée lors d'un changement 
de la structure de maillage (partitionnement, abandon de
mailles...).</td>
</tr>

<tr>
<td> **exit** </td>
<td>Appel du point d'entrée avant la sortie
du code : fin de simulation, arrêt avant reprise...</td>
</tr>

</table>

L'élément `<entry-points>` contient une liste de points d'entrée
nommés *nom_du_module.nom_du_point_d'entrée*. La valeur de l'attribut
`where` de chaque point d'entrée (faite dans \ref
arcanedoc_module_desc "le descripteur de module") doit être compatible
avec celle de l'attribut `where` du bloc `<entry-points>`. Dans un
bloc *init* ne peuvent être présent que des points d'entrée dont
l'attribut `where` vaut **init**, **start-init**,
**continue-init**. Pour les autres blocs (**compute-loop**,
**restore**, **on-mesh-changed**, **exit**), les valeurs des 2
attributs `where` doivent être identiques.
