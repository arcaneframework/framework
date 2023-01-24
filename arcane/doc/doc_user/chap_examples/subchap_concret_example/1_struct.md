# Structure générale {#arcanedoc_examples_concret_example_struct}

[TOC]

## Quicksilver Arcane Mini-App (QAMA) {#arcanedoc_examples_concret_example_struct_qama}

QAMA est une mini-application de transport de particule Monte-Carlo.
Cette application a été écrite à partir de Quicksilver, une mini-app
écrite par le LLNL mais avec l'utilisation des ressources fourni par
le framework %Arcane.

Voici un schéma représentant la structure de Quicksilver (disponible ici : https://github.com/arcaneframework/arcane-benchs) :

\image html QAMA_schema.jpg

On peut trouver (en autres) les éléments suivants :
- 3 modules nommés "QS", "SamplingMC" et "TrackingMC".
- 1 service "singleton" nommé "RNG".

Dans chaque module, nous avons les trois fichiers habituels :
- un header (.h)
- un fichier source (.cc)
- un fichier contenant les options du jeu de données (.axl)

L'interface du service RNG est une interface de service incluse
dans le framework %Arcane. Dans Quicksilver, on utilise notre propre implémentation.

\warning
Certains services, comme `BasicParticleExchanger` implémentant l'interface 
`Arcane::IParticleExchanger` ou `SimpleCsvOutput` implémentant l'interface
`Arcane::ISimpleTableOutput` sont des services inclus directement dans le framework %Arcane (c'est pour
cela qu'ils sont en pointillés).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_concret_example
</span>
<span class="next_section_button">
\ref arcanedoc_examples_concret_example_config
</span>
</div>
