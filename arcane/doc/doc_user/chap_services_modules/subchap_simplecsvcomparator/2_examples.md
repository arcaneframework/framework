# Exemples : généralités {#arcanedoc_services_modules_simplecsvcomparator_examples}

[TOC]

Dans les chapitres suivants, quelques exemples simples vous seront présentés.

Les 3 exemples présentés dans ce sous-chapitre sont fonctionnels et se trouvent dans le dossier :
`framework/arcane/samples_build/samples/simple_csv_comparator/`.

À noter également que ces exemples fonctionneront quelque soit l'implémentation de \arcane{ISimpleTableComparator}
(juste à changer l'implémentation dans `.config` (pour le mode singleton) ou dans les `.arc`).

Ces exemples ont des structures en communs : trois points d'entrée (`initModule`, `loopModule`, `endModule`) représentant
trois types de points d'entrée (`start-init`, `compute-loop`, `exit`) (au cas où : \ref arcanedoc_core_types_axl_entrypoint)
et aucunes variables.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_usage
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example1
</span>
</div>
