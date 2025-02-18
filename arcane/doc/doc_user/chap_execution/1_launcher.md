# Lancement d'un calcul {#arcanedoc_execution_launcher}

<!-- [TOC] -->

Il existe deux mécanismes pour exécuter un code avec %Arcane:

1. le mécanisme avec boucle en temps qui est le mécanisme classique
  disponible depuis les premières versions de %Arcane.
2. l'exécution directe

Pour les deux mécanismes, il faut utiliser la classe
Arcane::ArcaneLauncher. Cette classe permet de spécifier les
paramètres d'exécution. Toutes les méthodes de cette classe sont statiques

La première chose à faire est d'appeler la méthode
Arcane::ArcaneLauncher::init() pour spécifier à %Arcane les paramètres
d'exécution. Cela permet d'analyser automatiquement certaines valeurs
de la ligne de commande (comme le niveau de verbosité, le nom du
répertoire de sortie, ...)

```cpp
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(Arcane::CommandLineArguments(&argc,&argv));
  ...
}
```

Il existe deux classes permettant de spécifier les paramètres
d'exécution : Arcane::ApplicationInfo et
Arcane::ApplicationBuildInfo.

\note Ces deux classes existent pour des raisons de compatiblitées avec
le code existant. À terme, seule la classe
Arcane::ApplicationBuildInfo restera et c'est donc cette dernière
qu'il faut utiliser.

Les instances statiques de ces deux classes peuvent être récupérées
via les méthodes Arcane::ArcaneLauncher::applicationInfo() et
Arcane::ArcaneLauncher::applicationBuildInfo(). L'exemple suivant
montre comment changer le nom et la version du code et le répertoire
par défaut pour les sorties :

```cpp
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(Arcane::CommandLineArguments(&argc,&argv));
  Arcane::ApplicationBuildInfo& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName("ArcaneTest");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  app_build_info.setOutputDirectory("test_output");
  ...
}
```

Une fois l'initialisation effectuée, il est possible de lancer
l'exécution du code via l'appel à Arcane::ArcaneLauncher::run() :

```cpp
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(Arcane::CommandLineArguments(&argc,&argv));
  Arcane::ApplicationBuildInfo& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName("ArcaneTest");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  app_build_info.setOutputDirectory("test_output");
  return ArcaneLauncher::run();
}
```

## Initialisation de MPI {#arcanedoc_execution_launcher_mpi}

L'utilisation de MPI nécessite de faire un appel à la méthode
`MPI_Init_thread()` de la bibliothèque MPI. Si %Arcane est compilé avec le
support de MPI, alors la détection de MPI et l'appel à `MPI_Init_thread()` est
fait automatiquement par %Arcane. Le niveau de support des threads
utilisé pour l'appel à `MPI_Init_thread()` dépend des options telles que
le nombre de tâches ou de sous-domaines locaux en mode hybride qu'on
souhaite utiliser.

Il est néanmoins possible pour le code d'initialiser lui-même MPI s'il
le souhaite. Pour cela, il doit appeler la méthode `MPI_Init_thread()`
avant l'appel à Arcane::ArcaneLauncher::run().

\note Même si l'exécutable est utilisé en séquentiel (c'est-à-dire
sans passer par une commande telle que `mpiexec ...`), %Arcane tente
d'initialiser MPI. Cela est nécessaire car certaines bibliothèques
(par exemple les solveurs linéaires) ont besoin que MPI soit
initialisé dans tous les cas. Il est possible de modifier ce
comportement en spécifiant explicitement le service de parallélisme
souhaité (TODO faire doc).

\warning Il faut faire attention à bien utiliser l'exécutable
`mpiexec` qui correspond à la version de MPI avec laquelle %Arcane a
été compilé sinon on va lancer *N* fois l'exécution séquentielle.

## Exécution du code {#arcanedoc_execution_launcher_exec}

La méthode Arcane::ArcaneLauncher::run() permet de lancer l'exécution
du code. Cette méthode possède trois surcharges :

1. L'appel sans argument (Arcane::ArcaneLauncher::run()) pour lancer l'exécution classique
   en utilisant une boucle en temps (voir \ref
   arcanedoc_core_types_codeconfig). Ce mécanisme est à privilégier car elle
   permet de disposer de toutes les fonctionnalités de %Arcane. La
   page \ref arcanedoc_execution_launcher montre un exemple minimal de ce
   type d'utilisation.
2. Arcane::ArcaneLauncher::run(std::function<int(DirectSubDomainExecutionContext&)>
   func) pour exécuter le code spécifié par la \a func après
   l'initialisation et la création des sous-domaines. La page
   \ref arcanedoc_general_direct_execution montre un exemple d'exécution directe.
3. Arcane::ArcaneLauncher::run(std::function<int(DirectExecutionContext&)>
   func) pour exécuter uniquement en **séquentiel** le code spécifié par la \a
   func. Ce mécanisme est à utiliser si on souhaite par exemple faire des tests
   unitaires simples sans avoir de sous-domaine (Arcane::ISubDomain*).
   application sans jeu de données ni boucle en temps.

## Options de la ligne de commande {#arcanedoc_execution_launcher_options}

%Arcane interprète les options de la ligne de commande qui commencent
par `-A`. Par exemple, pour changer le niveau de verbosité, il suffit
de spécifier l'option `-A,VerbosityLevel=3` dans la ligne de commande.

Les options sont interprétées lors de l'appel à
Arcane::ArcaneLauncher::init() et les valeurs de
ArcaneLauncher::applicationBuildInfo() sont automatiquement remplies
avec ces options. Il est cependant possible de les surcharger si
nécessaire.

\remark Il est aussi possible de modifier le jeu de données avec des
options de la ligne de commande. Cette possibilité est abordée à la
page \ref arcanedoc_execution_commandlineargs.

Les options disponibles sont :

<table>
<tr>
<th>Option</th>
<th>Variable d'environnement</th>
<th>Type</th>
<th>Défaut</th>
<th>Description</th>
</tr>

<tr>
<td>T</td>
<td>ARCANE_NB_TASK</td>
<td>Int32</td>
<td>1</td>
<td>Nombre de tâches concurrentes à exécuter</td>
</tr>

<tr>
<td>S</td>
<td>ARCANE_NB_THREAD (Cette variable d'environnement est obsolète)</td>
<td>Int32</td>
<td></td>
<td>Nombre de sous-domaines en mémoire partagée</td>
</tr>

<tr>
<td>R</td>
<td>ARCANE_NB_REPLICATION (Cette variable d'environnement est obsolète)</td>
<td>Int32</td>
<td>1</td>
<td>Nombre de sous-domaines répliqués</td>
</tr>

<tr>
<td>P</td>
<td>ARCANE_NB_SUB_DOMAIN (Cette variable d'environnement est obsolète)</td>
<td>Int32</td>
<td></td>
<td>Nombre de processus à utiliser pour les sous-domaines. Cette
valeur est normalement calculée automatiquement en fonction des
paramètres MPI. Elle n'est utile que si on souhaite utiliser moins de
processus pour le partitionnement de domaine que ceux alloués pour le
calcul.
</td>
</tr>

<tr>
<td>AcceleratorRuntime</td>
<td></td>
<td>string</td>
<td></td>
<td>Runtime accélérateur à utiliser. Les deux valeurs possibles sont
`cuda` ou `hip`. Il faut avoir compiler %Arcane avec le support des
accélérateurs pour que cette option soit accessible.
</td>
</tr>

<tr>
<td>MaxIteration</td>
<td></td>
<td>Int32</td>
<td></td>
<td>Nombre maximum d'itérations à effectuer pour l'exécution. Si le
nombre d'itérations spécifié par cette variable est atteint, le calcul
s'arrête.
</td>
</tr>

<tr>
<td>OutputLevel</td>
<td>ARCANE_OUTPUT_LEVEL</td>
<td>Int32</td>
<td>3</td>
<td>Niveau de verbosité des messages sur la sortie standard.</td>
</tr>

<tr>
<td>VerbosityLevel</td>
<td>ARCANE_VERBOSITY_LEVEL</td>
<td>Int32</td>
<td>3</td>
<td>Niveau de verbosité des messages pour les sorties listings
fichiers. Si l'option `OutputLevel` n'est pas spécifiée, cette option
est aussi utilisée pour les sorties standards.
</td>
</tr>

<tr>
<td>MinimalVerbosityLevel</td>
<td></td>
<td>Int32</td>
<td></td>
<td>Niveau de verbosité minimal. Si spécifié, les appels explicites
dans le code pour modifier la verbosité (via
Arccore::ITraceMng::setVerbosityLevel()) ne pourront pas descendre en
dessous de ce niveau de verbosité minimal. Ce mécanisme est surtout
utilisé en débug pour garantir l'affichage des messages.
</td>
</tr>

<tr>
<td>MasterHasOutputFile</td>
<td>ARCANE_MASTER_HAS_OUTPUT_FILE</td>
<td>Bool</td>
<td>False</td>
<td>Indique si le processus maitre (en général le processus 0) écrit
le listing dans un fichier en plus de la sortie standard</td>
</tr>

<tr>
<td>OutputDirectory</td>
<td>ARCANE_OUTPUT_DIRECTORY</td>
<td>String</td>
<td>.</td>
<td>Répertoire de base pour les fichiers générés (listings, logs,
courbes, dépouillement, ...). Cette valeur est celle retournée par
Arcane::ISubDomain::exportDirectory().
</td>
</tr>

<tr>
<td>CaseDatasetFileName</td>
<td></td>
<td>String</td>
<td></td>
<td>Nom du fichier du jeu de données. Si non spécifié et requis, le
dernier argument de la ligne de commande est considéré comme le nom de
fichier du jeu données.
</td>
</tr>

<tr>
<td>ThreadBindingStrategy</td>
<td>ARCANE_THREAD_BINDING_STRATEGY</td>
<td>String</td>
<td></td>
<td>Stratégie de punnaisage des threads. Cela fonctionne uniquement si
%Arcane est compilé avec la bibliothèque 'hwloc'. Par défaut aucun
binding n'est effectué. Le seul mode disponible est 'Simple' qui
alloue les threads suivant un mécanisme round-robin.

NOTE: ce mécanisme de punnaisage est en cours de développement et il
est possible qu'il ne fonctionne pas de manière optimale dans tous les
cas
</td>
</tr>

<tr>
<td>ParallelLoopGrainSize</td>
<td></td>
<td>Int32</td>
<td></td>
<td>Taille de grain pour les boucles parallèles multi-threadées. Si
positionné, indique le nombre d'éléments de chaque bloc qui décompose une boucle
multi-threadée (à partir de la version 3.8).
</td>
</tr>

<tr>
<td>ParallelLoopPartitioner</td>
<td></td>
<td>String</td>
<td></td>
<td>Choix du partitionneur pour les boucles parallèles
multi-threadées. Les valeurs possibles sont `auto`, `static` ou
`deterministic` (à partir de la version 3.8).
</td>
</tr>

</table>

## Choix du gestionnaire d'échange de message {#arcanedoc_execution_launcher_exchange}

Le gestionnaire d'échange de message (Arcane::IParallelSuperMng) est
choisi lors du lancement du calcul.  %Arcane fournit les gestionnaires suivants :

- MpiParallelSuperMng
- SequentialParallelSuperMng
- MpiSequentialParallelSuperMng
- SharedMemoryParallelSuperMng
- HybridParallelSuperMng

En général, %Arcane choisit
automatiquement le gestionnaire en fonction des paramètres utilisés
pour lancer le calcul mais il est possible de spécifier explicitement
le gestionnaire à utiliser en positionnant la variable d'environnement (obsolète)
`ARCANE_PARALLEL_SERVICE` ou en spécifiant l'option
`MessagePassingService` dans la ligne de commande avec une des valeurs
ci-dessus (sans le suffixe `ParallelSuperMng`, donc par exemple `Mpi`,
`Sequential`, `MpiSequential`, ...).

Le choix automatique du gestionnaire est fait comme suit :

<table>
<tr>
<th>Ligne de commande</th>
<th>Gestionnaire utilisé</th>
<th>Description</th>
</tr>
<tr>
<td>`./a.out ...`</td>
<td>`MpiSequentialParallelSuperMng` ou `SequentialParallelSuperMng`</td>
<td>`MpiSequentialParallelSuperMng` si %Arcane a été
compilé avec MPI, `SequentialParallelSuperMng` sinon. La différence
entre les deux est que le premier initialise MPI afin que les
communicateurs tels que `MPI_COMM_WORLD` puissent être utilisés
</td>
</tr>

<tr>
<td>`mpiexec -n $N ./a.out ...`</td>
<td>`MpiParallelSuperMng`</td>
<td>$N processus, 1 sous-domaine par processus</td>
</tr>

<tr>
<td>`./a.out -A,S=$S ...`</td>
<td>`SharedMemoryParallelSuperMng`</td>
<td>1 processus, $S sous-domaines par processus. La communication
entre les sous-domaines se fait par échange de message en mémoire partagée.
</td>
</tr>

<tr>
<td>`mpiexec -n $N ./a.out -A,S=$S ...`</td>
<td>`HybridParallelSuperMng`</td>
<td>$N processus, $S sous-domaines par processus
soit au total $N * $S sous-domaines.
</td>
</tr>

</table>

Voici quelques exemples de lancement :

```sh
# lancement séquentiel du jeu de données 'Test.arc'
a.out Test.arc

# lancement avec 4 sous-domaines MPI
mpiexec -n 4 a.out Test.arc

# lancement avec 4 sous-domaines en mode mémoire partagée
a.out -A,S=4 Test.arc

# lancement avec 12 sous-domaines et 4 processus (4 sous-domaines en
# mémoire partagée par processus)
mpiexec -n 3 -c 4 a.out -A,S=4 Test.arc

# lancement avec le runtime accélérateur CUDA.
a.out -A,AcceleratorRuntime=cuda Test.arc
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution
</span>
<span class="next_section_button">
\ref arcanedoc_execution_direct_execution
</span>
</div>
