# Lancement direct d'une exécution {#arcanedoc_execution_direct_execution}

[TOC]

Il est possible d'utiliser %Arcane sans passer par les mécanismes
utilisant les modules et la boucle en temps. Cela peut être utile pour
des codes très simples ou des utilitaires mais ce mécanisme est
déconseillé pour les gros codes de calcul car il ne permet pas
d'accéder automatiquement à l'ensemble des fonctionnalités de %Arcane
comme par exemple l'équilibrage de charge, les protections/reprises ou
les modules (même si ces mécanismes restent accessibles manuellement).

Il existe deux manière de lancer le mode autonome:

- le mode avec support des accélérateurs. Dans ce mode seule l'API
  accélérateur et les classes utilitaires de %Arcane sont disponibles.
  La page \ref arcanedoc_parallel_accelerator_standalone décrit
  comment utiliser ce mode.
- le mode avec sous-domaine. Ce mode permet d'accéder manuellement à
  la plupart des fonctionnalités d'%Arcane comme le maillage, le
  dépouillement ou l'équilibrage de charge.

Les deux exemples `standalone_subdomain` et `standalone_accelerator`
montrent comment utiliser ces mécanismes.

La page \ref arcanedoc_execution_launcher explique comment fournir les
paramètres pour initialiser %Arcane.

## Mode Sous-domaine Autonome {#arcanedoc_parallel_accelerator_standalone_subdomain}

Ce mode permet de piloter manuellement la plupart des fonctionnalités
d'%Arcane comme les maillages et le dépouillement. Pour utiliser ce
mode, il suffit d'utiliser la méthode de classe
\arcane{ArcaneLauncher::createStandaloneSubDomain()} après avoir
initialiser %Arcane :

```cpp
Arcane::String case_file_name = {};
Arcane::ArcaneLauncher::init(Arcane::CommandLineArguments(&argc, &argv));
Arcane::StandaloneSubDomain sub_domain(Arcane::ArcaneLauncher::createStandaloneSubDomain(case_file_name));
```

Il est possible de spécifier un nom de fichier pour le jeu de
données. Dans ce cas, si ce fichier comporte des maillages, ces
derniers seront automatiquement créés lors de la création du
sous-domaine.

L'instance `sub_domain` doit rester valide tant qu'on souhaite utiliser
le sous-domaine. Il est donc préférable de la définir dans le
`main()` du code.

\warning Un seul appel à \arcane{ArcaneLauncher::createStandaloneSubDomain} est autorisé.

Par exemple, le code suivant permet de lire un maillage, d'afficher le
nombre de mailles, de calculer et d'afficher les coordonnées des
centres des mailles.

\snippet standalone_subdomain/main.cc StandaloneSubDomainFull

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_launcher
</span>
<span class="next_section_button">
\ref arcanedoc_execution_env_variables
</span>
</div>
