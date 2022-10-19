# Service SimpleCsvOutput {#arcanedoc_services_modules_simplecsvoutput}

[TOC]

- Documentation générée du service disponible ici : \ref axldoc_service_SimpleCsvOutput_arcane_std
- Documentation de l'interface implémentée par ce service : \arcane{ISimpleTableOutput}

\warning
L'interface n'est pas encore figée. Elle peut donc encore évoluer.

____

Ce service permet de créer un tableau 2D de valeur avec des lignes et des colonnes nommées. Aujourd'hui, le format de fichier en sortie est le format CSV.
Ce service peut être utilisé comme service classique à définir dans l'AXL d'un module ou comme singleton pour avoir une instance unique pour tous les modules.

Il suffit de créer une ou plusieurs lignes et une ou plusieurs colonnes, puis d'attribuer des valeurs à chaque [ligne;colonne] et enfin d'appeler
la méthode writeFile() pour générer un fichier.csv.

Exemple de fichier .csv :
```csv
Results_Example3;Iteration 1;Iteration 2;Iteration 3;
Nb de Fissions;36;0;85;
Nb de Collisions;29;84;21;
```
Sous Excel (ou un autre tableur), on obtient ce tableau :
| Results_Example3 | Iteration 1 | Iteration 2 | Iteration 3 |
|------------------|-------------|-------------|-------------|
| Nb de Fissions   | 36          | 0           | 85          |
| Nb de Collisions | 29          | 84          | 21          |

Ce sous-chapitre permet d'introduire ce service. Tous les cas d'utilisations ne seront pas abordés,
il est donc recommandé d'aller voir la documentation de l'interface \arcane{ISimpleTableOutput}
pour pouvoir exploiter pleinement ce service (notamment l'aspect gestion du multi-processus (TODO : faire exemple)).

<br>

Sommaire de ce sous-chapitre :

1. \subpage arcanedoc_services_modules_simplecsvoutput_usage <br>
  todo

2. \subpage arcanedoc_services_modules_simplecsvoutput_examples <br>
  todo

3. \subpage arcanedoc_services_modules_simplecsvoutput_example1 <br>
  todo

4. \subpage arcanedoc_services_modules_simplecsvoutput_example2 <br>
  todo

____


<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_usage
</span>
</div>
