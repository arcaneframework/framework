# Exemples : généralités {#arcanedoc_services_modules_simplecsvoutput_examples}

[TOC]

Dans les chapitres suivants, quelques exemples simples vous seront présentés.

Lors de la description des tableaux, pour souligner le nombre de lignes
et de colonnes, la ligne et la colonne contenant les titres sont omises
dans les décomptes, tout comme dans le service.

Les 6 exemples présentés dans ce sous-chapitre sont fonctionnels et se trouvent dans le dossier :
`framework/arcane/samples_build/samples/simple_csv_output/`.

À noter également que ces exemples fonctionneront quelque soit l'implémentation de \arcane{ISimpleTableOutput}
(juste à changer l'implémentation dans `.config` (pour le mode singleton) ou dans les `.axl`).

Ces exemples ont des structures en communs : trois points d'entrée (`initModule`, `loopModule`, `endModule`) représentant
trois types de points d'entrée (`start-init`, `compute-loop`, `exit`) (au cas où : \ref arcanedoc_core_types_axl_entrypoint)
et aucunes variables.  
Les options varient pour les exemples 1, 2 et 3-6.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_usage
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example1
</span>
</div>
