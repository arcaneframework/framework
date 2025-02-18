# Utilisation des traces {#arcanedoc_execution_traces}

[TOC]

## Introduction {#arcanedoc_execution_traces_intro}

%Arcane fournit une classe utilitaire (Arcane::TraceAccessor) pour
afficher des traces dans les modules. Cette classe permet de gérer
plusieurs types de traces : informations, erreurs, ...

Si dans le descripteur de module, l'attribut `parent-name` de l'élément
`module` vaut `Arcane::BasicModule` (le défaut), les traces sont
automatiquement disponibles.

Les traces s'utilisent comme des flots classiques en C++, grâce à
l'operateur <<.

Par exemple, pour afficher une trace d'information :

```cpp
Arcane::TraceAccessor::info() << "Ceci est un message d'information";
```

Tous les types C++ qui disposent de l'opérateur `operator<<()` peuvent
être tracés. Par exemple :

```cpp
int z = 3;
Arcane::TraceAccessor::info() << "z vaut " << z;
```

A noter qu'un retour chariot est effectué automatiquement entre
chaque message. Par conséquent l'ajout d'un retour chariot en fin de trace
provoque un saut de ligne.

## Catégories de trace {#arcanedoc_execution_traces_class}

Les méthodes de trace sont :
- \b Arcane::TraceAccessor::info() pour les traces d'informations,
- \b Arcane::TraceAccessor::debug() pour les traces de debug,
- \b Arcane::TraceAccessor::log() pour les traces de log,
- \b Arcane::TraceAccessor::warning() pour les traces d'avertissement,
- \b Arcane::TraceAccessor::error() pour les traces d'erreur,
- \b Arcane::TraceAccessor::fatal() pour les traces d'erreur fatale,
  ce qui arrête l'exécution. Il est aussi possible d'utiliser la macro
  ARCANE_FATAL() pour obtenir le même comportement. L'avantage de la
  macro est qu'elle indique explictement au compilateur qu'on lance
  une exception de type Arcane::FatalErrorException() ce qui peut
  permettre d'éviter des avertissements de compilation.

Les traces d'avertissement ou d'erreur
(Arcane::TraceAccessor::warning(), Arcane::TraceAccessor::error() et
Arcane::TraceAccessor::fatal()) sont toujours affichées. Pour les
traces d'informations (Arcane::TraceAccessor::info()) et de débug
(Arcane::TraceAccessor::debug()), le comportement dépend de
l'exécution séquentielle ou parallèle et si ARCANE est compilée en
mode débug ou optimisé:
- en mode optimisé, les traces de debug ne sont jamais actives. De
plus, la méthode `debug()` est remplacée par une méthode vide ce qui
fait qu'elle ne prend aucune ressource CPU.
- en mode optimisé, par défaut, les traces d'informations ne sont
affichées que par le sous-domaine 0. Ce comportement est configurable
(voir section \ref arcanedoc_execution_traces_config).
- en mode débug, les traces du sous-domaine 0 s'affichent sur la
sortie standard. Les traces des autres sous-domaines sont écrites
dans un fichier de nom 'output%n', où '%n' est le numéro du
sous-domaine.

Les traces de log sont écrites dans un fichier dans le répertoire
'listing', sous le nom 'log.%n', avec '%n' le numéro du
sous-domaine.

Il existe 4 méthodes pour la gestion parallèle des traces :
- \b Arcane::TraceAccessor::pinfo() pour les traces d'informations,
- \b Arcane::TraceAccessor::pwarning() pour les traces d'avertissement,
- \b Arcane::TraceAccessor::perror() pour les traces d'erreur,
- \b Arcane::TraceAccessor::pfatal() pour les traces d'erreur fatale, ce qui arrête l'exécution.

Pour pinfo(), chaque sous-domaine affiche le message. Pour les
autres (Arcane::TraceAccessor::pwarning(),
Arcane::TraceAccessor::perror() et Arcane::TraceAccessor::pfatal()),
cela signifie que chaque sous-domaine appelle cette méthode (opération
collective), et donc une seule trace sera  affichée. Ces traces
parallèles peuvent par exemple être utiles lorsqu'on est certain que
l'erreur produite le sera par tous les processeurs, par exemple, une
erreur dans le jeu de données. Il faut prendre soin que tous les
sous-domaines appellent les méthodes collectives, car cela peut conduire à un
blocage du code dans le cas contraire.

Il faut noter qu'en cas d'appel à la méthode Arcane::TraceAccessor::fatal() en parallèle,
les processus sont en général tués sans ménagement. Avec Arcane::TraceAccessor::pfatal(),
il est possible d'arrêter le code proprement puisque chaque
sous-domaine génère l'erreur.

Il existe trois niveaux de traces pour la catégorie \c debug : 
Arccore::Trace::Low, \a Arccore::Trace::Medium et Arccore::Trace::High. Le niveau par défaut
est \a Arccore::Trace::Medium.

```cpp
Arcane::TraceAccessor::debug(Arccore::Trace::Medium) << "Trace debug moyen"
Arcane::TraceAccessor::debug() << "Trace debug moyen"
Arcane::TraceAccessor::debug(Arccore::Trace::Low)    << "Trace debug affiché dès que le mode debug est utilisé"
```

## Configuration des traces {#arcanedoc_execution_traces_config}

Il est possible de configurer le niveau de debug souhaité et
l'utilisation des traces d'informations pour chaque module
dans le fichier de configuration de ARCANE. Ce fichier de configuration 
utilisateur permet de modifier le comportement
par défaut de certains éléments de l'architecture tels que
l'affichage des traces. Il est nommé <em>config.xml</em> et 
se trouve dans le répertoire <tt>.arcane</tt> du compte de l'utilisateur
qui lance l'exécution.

La configuration se fait avec les attributs
\c name, \c info et \c debug de l'élément \c trace-module. 
Cet élément doit être fils de l'élément \c traces.

- \b name spécifie le nom du module concerné
- \b info vaut \e true s'il faut afficher les traces d'informations,
\e false sinon.
- \b debug vaut \e none, \e low, \e medium ou \e high suivant le niveau de debug souhaité.
  Les traces de debug d'un niveau supérieur à celui demandé ne sont
  pas affichées. Le niveau \e high correspond à toutes les traces.

Voici un exemple de fichier : 

```xml
<?xml version="1.0" encoding="ISO-8859-1" ?>
<arcane-config>
  <traces>
    <trace-class name="*" info="true" debug="none" />
    <trace-class name="Hydro" info="true" debug="medium" />
    <trace-class name="ParallelMng" info="true" print-class-name="false" print-elapsed-time="true" />
  </traces>
</arcane-config>
```

Dans l'exemple, l'utilisateur demande à ce que les traces d'informations 
pour tous les modules soient par défaut activés et pas les traces de debug.
Pour le module Hydro, sont affichées les traces d'informations et les traces 
de debug jusqu'au niveau \e medium.
Pour la classe de message ParallelMng, on affiche les infos et le
temps écoulé mais pas le nom de la classe du message (c'est-à-dire
le début de la ligne '*I-ParallelMng'.

\note Quelle que soit la configuration, les traces de débug ne
sont pas disponibles en version optimisée complète.

Il est possible de changer dynamiquement les informations d'une
classe de message. Par exemple le code suivant permet depuis un module ou service de
changer le niveau de verbosité et d'afficher le temps écoulé
mais pas le nom de la classe de message :

```cpp
Arcane::ITraceMng* tm = traceMng();
Arcane::TraceClassConfig tcc = tm->classConfig("MyTest");
tcc.setFlags(Trace::PF_ElapsedTime|Trace::PF_NoClassName);
tcc.setVerboseLevel(4);
tm->setClassConfig("MyTest",tcc);
```


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_env_variables
</span>
<span class="next_section_button">
\ref arcanedoc_execution_commandlineargs
</span>
</div>

