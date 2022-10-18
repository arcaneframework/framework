# Types fondamentaux {#arcanedoc_core_types}

Il existe 4 types fondamentaux dans %Arcane, qui correspondent aux
notions de **Module**, **Service**, **Variable** et **Point d'entrée**.  
Ces 4 types fondamentaux sont présents dans trois types de fichiers propres
à %Arcane : **Fichier AXL** (Descripteur de module/service), **Fichier ARC** 
(Jeu de données) et **Fichier config**.

Pour une description sommaire de ces notions, se reporter au chapitre \ref arcanedoc_getting_started.

Voici un schéma de code %Arcane simple, avec un module et deux services. Les deux services partagant
une interface commune.

\image html code_schema.svg

Les différentes parties de ce chapitre devrait vous permettre de comprendre ce schéma
(mise à part les fichiers main.cc et CMakeLists.txt qui sont expliqués dans le chapitre 
\ref arcanedoc_execution).

<br>

Sommaire de ce chapitre :
1. \subpage arcanedoc_core_types_module <br>
  Présente la notion de module dans %Arcane.

2. \subpage arcanedoc_core_types_service <br>
  Présente la notion de service dans %Arcane.

3. \subpage arcanedoc_core_types_axl <br>
  Présente tout ce qu'il y a à savoir des descripteurs de module/service 
  (représentés par les fichiers ayant l'extension .axl).
  C'est dans ce sous-chapitre que sont présentés les notions de \ref arcanedoc_core_types_axl_variable
  et de \ref arcanedoc_core_types_axl_entrypoint.

4. \subpage arcanedoc_core_types_casefile <br>
  Présente la syntaxe du jeu de données
  (représenté par les fichiers ayant l'extension .arc).

5. \subpage arcanedoc_core_types_codeconfig <br>
  Présente le fichier de configuration global du code.

6. \subpage arcanedoc_core_types_timeloop <br>
  Décrit la notion de boucle en temps.

7. \subpage arcanedoc_core_types_array_usage <br>
  Décrit l'utilisation des types tableaux.

____

<div class="section_buttons">
<span class="next_section_button">
\ref arcanedoc_core_types_module
</span>
</div>