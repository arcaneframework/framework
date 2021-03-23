La boucle en temps {#arcanedoc_timeloop}
================== 

Introduction {#arcanedoc_timeloop_intro}
============

Un code de simulation construit avec la plate-forme %Arcane
est composé d'un ensemble de \ref arcanedoc_module "modules numériques".
Ces modules contiennent des \ref arcanedoc_variable "variables" et 
des \ref arcanedoc_entrypoint "points d'entrée".

L'enchaînement des calculs du code est décrit par une succession 
de points d'entrée, la boucle en temps. Les boucles
en temps sont définies dans le fichier de configuration de
l'application (voir \ref arcanedoc_codeconfig)

Lors de l'exécution d'un cas, la boucle en temps souhaitée pour
le cas est choisie dans le jeu de données. Changer de boucle
en temps dans le jeu de données de la simulation ou modifier
le fichier de description des boucles en temps ne nécessite 
pas de recompilation.

## Utilisation ## {#arcanedoc_timeloop_use}

Pour la réalisation d'une simulation, il faut choisir une boucle
en temps parmi celles définies dans le fichier précédent de configuration. Cette
sélection s'effectue dans l'élément *timeloop* de l'élément
*arcane* du jeu de données. La boucle en temps est identifiée
par son nom.

~~~~~~~~~~~~~~~~{.xml}
<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="MicroHydro" xml:lang="en">
 <arcane>
  <title>Exemple Arcane d'un module Hydro très, très simplifié</title>
  <timeloop>MicroHydroLoop</timeloop>
 </arcane>
 ...
</case>
~~~~~~~~~~~~~~~~


