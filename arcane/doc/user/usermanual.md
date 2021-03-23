Manuel utilisateur de la plate-forme ARCANE {#arcanedoc_usermanual}
=========================

\mainpage %Arcane

Cette page contient la documentation pour les utilisateurs de la plate-forme %Arcane.

La page \subpage arcanedoc_changelog présente la liste des nouveautés de chaque version.

La documentation accessible depuis cette page s'adresse aux personnes
désirant connaitre les fonctionnalités de la plate-forme %Arcane ou
souhaitant développer des modules ou services capables de s'exécuter
sur la plate-forme %Arcane.

Toutes les personnes développant des modules ou services pour %Arcane
doivent connaitre au minimum les documents \ref arcane_overview et
\ref arcanedoc_core. Ensuite, pour avoir le détail des opérations
d'une classe, par exemple pour connaitre l'ensemble des méthodes de la
classe Arcane::Cell, il faut consulter la documentation en ligne des
codes sources de %Arcane.

Les documents disponibles sont:

- \subpage arcane_overview : présente briévement les principes et 
  les types de base de %Arcane.
- \subpage arcanedoc_core : présente les notions clefs de %Arcane (modules,
  variables, points d'entrée et services).
- \subpage arcanedoc_tutorial : présente un didacticiel de %Arcane.
- \subpage arcanedoc_codingrules : présente les règles de codage utilisées
   dans %Arcane. Pour que les modules développés sur la plate-forme aient
   une certaine homogénéité, il est intéressant de suivre ces règles.
- \subpage arcanedoc_launcher : Lancement d'un calcul
- \subpage arcanedoc_caseoptions : explique comment paramètrer les modules
   avec des options utilisateurs fournies dans le jeu de données.
- \subpage arcanedoc_codeconfig : présente le fichier de configuration d'un exécutable
   réalisé avec la plate-forme %Arcane. Ce fichier contient entre autre la description des
   boucles en temps disponibles.
- \subpage arcanedoc_casefile : présente la syntaxe du jeu de données.
- \subpage arcanedoc_timeloop : décrit la notion de boucle en temps.
- \subpage arcanedoc_parallel : présente la manière dont %Arcane prend en charge le parallèlisme par partionnement de domaine.
- \subpage arcanedoc_concurrency : décrite la notion de concurrence et la parallélisation au niveau des boucles
- \subpage arcanedoc_simd : décrit l'utilisation de la vectorisation (SIMD).
- \subpage arcanedoc_traces : décrit comment afficher des traces pendant les
   exécutions et comment paramétrer leur affichage.
- \subpage arcanedoc_dotnet : décrit comment utiliser et étendre les fonctionnalités de %Arcane en `C#`.
- \subpage arcanedoc_user_unit_tests : décrit comment réaliser des tests unitaires pour les modules et services
- \subpage arcanedoc_material
- \subpage arcanedoc_materialloop
- \subpage arcanedoc_cartesianmesh
- \subpage arcanedoc_cea_geometric

Autres informations:

- \subpage arcanedoc_itemtype : décrit les types de maille classiques.
- \subpage arcanedoc_env_variables : liste des variables d'environnement permettant de modifier le comportement de %Arcane
- \subpage arcanedoc_check_memory : détection des problèmes mémoire.
- \subpage arcanedoc_compare_bittobit : comparaison bit à bit de deux exécutions
- \subpage arcanedoc_profiling_toc : décrit les mécanismes disponibles pour l'analyse de performance
- \subpage arcanedoc_array_usage : décrit l'utilisation des types tableaux.
- \subpage arcanedoc_mesh_loadbalance : décrit l'utilisation du mécanisme d'équilibrage de charge sur le maillage.
- \subpage arcanedoc_connectivity_internal décrit le nouveau mécanisme (à partir de la version 2.5) de gestion des connectivités des entités.

Nettoyage pour la v2.0:

- \subpage arcanedoc_cleanup_v2 décrit les modifications à effectuer pour préparer le passage à la version 2.0
