# Boucle en temps {#arcanedoc_core_types_timeloop}

[TOC]

Un code de simulation construit avec la plate-forme %Arcane
est composé d'un ensemble de \ref arcanedoc_core_types_module "modules numériques".
Ces modules contiennent des \ref arcanedoc_core_types_axl_variable "variables" et 
des \ref arcanedoc_core_types_axl_entrypoint "points d'entrée".

L'enchaînement des calculs du code est décrit par une succession 
de points d'entrée, la boucle en temps. Les boucles
en temps sont définies dans le fichier de configuration de
l'application (voir \ref arcanedoc_core_types_codeconfig)

Lors de l'exécution d'un cas, la boucle en temps souhaitée pour
le cas est choisie dans le jeu de données. Changer de boucle
en temps dans le jeu de données de la simulation ou modifier
le fichier de description des boucles en temps ne nécessite 
pas de recompilation.

## Utilisation {#arcanedoc_core_types_timeloop_use}

Pour la réalisation d'une simulation, il faut choisir une boucle
en temps parmi celles définies dans le fichier précédent de configuration. Cette
sélection s'effectue dans l'élément *timeloop* de l'élément
*arcane* du jeu de données. La boucle en temps est identifiée
par son nom.

```xml
<?xml version='1.0'?>
<case codeversion="1.0" codename="MicroHydro" xml:lang="en">
 <arcane>
  <title>Exemple Arcane d'un module Hydro très, très simplifié</title>
  <timeloop>MicroHydroLoop</timeloop>
 </arcane>
 ...
</case>
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_codeconfig
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_array_usage
</span>
</div>
