# Service SimpleCsvComparator {#arcanedoc_services_modules_simplecsvcomparator}

[TOC]

- Documentation générée du service disponible ici : \ref axldoc_service_SimpleCsvComparator_arcane_std
- Documentation de l'interface implémentée par ce service : \arcane{ISimpleTableComparator}

\warning
L'interface n'est pas encore figée. Elle peut donc encore évoluer.

____

\warning
Ce sous-chapitre a été pensé comme une suite au sous-chapitre \ref arcanedoc_services_modules_simplecsvoutput.

Ce service permet de comparer les valeurs de deux `SimpleTableInternal` entre eux.  
Lors d'un lancement de code ayant intégrer un service du type \arcane{ISimpleTableOutput},
il est possible de générer un fichier de référence (ou plusieurs, un par sous-domaine, si on le souhaite).

Puis, lors d'un lancement suivant, il est possible de comparer les valeurs du fichier de référence
généré précédemment avec les valeurs stockées dans le service du type \arcane{ISimpleTableOutput}
du lancement actuel.

Grâce au format CSV, il est aussi possible de visualiser et de modifier les valeurs de références,
si l'on souhaite.

Ce service peut être utilisé comme service classique à définir dans l'AXL d'un module ou comme 
singleton pour avoir une instance unique pour tous les modules.

Ce sous-chapitre permet d'introduire ce service. Tous les cas d'utilisations ne seront pas abordés,
il est donc recommandé d'aller voir la documentation de l'interface \arcane{ISimpleTableComparator}
pour pouvoir exploiter pleinement ce service.

<br>

Sommaire de ce sous-chapitre :

1. \subpage arcanedoc_services_modules_simplecsvcomparator_usage <br>
  Résume comment utiliser le service.

2. \subpage arcanedoc_services_modules_simplecsvcomparator_examples <br>
  Quelques généralités à lire avant d'attaquer les exemples.

3. \subpage arcanedoc_services_modules_simplecsvcomparator_example1 <br>
  Cet exemple simple introduit comment utiliser le service en mode singleton.

4. \subpage arcanedoc_services_modules_simplecsvcomparator_example2 <br>
  Cet exemple n'utilise pas le mode singleton.

5. \subpage arcanedoc_services_modules_simplecsvcomparator_example3 <br>
  Cet exemple mixe un `SimpleCsvOutput` singleton et un `SimpleCsvComparator`
  sans. Il y a aussi un exemple d'utilisation des expressions régulières.


____


<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_usage
</span>
</div>
