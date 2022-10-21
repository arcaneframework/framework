# Point d'entrée {#arcanedoc_core_types_axl_entrypoint}

[TOC]

Un point d'entrée est une méthode d'un module qui est référencée par %Arcane
et qui sert d'interface pour le module avec la boucle en temps. Un point d'entrée est décrit par
la classe interface \c IEntryPoint. Un point d'entrée est une méthode dont la signature est la
suivante, <b>T</b> étant le type de la classe du module:

```cpp
void T::func();
```

Un point d'entrée est caractérisé par:
- un nom
- une méthode de la classe associée.
- l'endroit où il peut être appelé (initialisation, boucle de calcul, ...).
Par défaut, un point d'entrée est appelé dans la boucle de calcul.

Les points d'entrée sont déclarés dans le descripteur de module. Par exemple, pour le 
module \c Test, on déclare 2 points d'entrée pouvant ensuite être appelés dans 
la boucle en temps par les noms <b>DumpConnection</b> ou <b>TestPressureSync</b> :

```xml
<module name="Test" version="1.0">
	<name lang="fr">Test</name>

	<description>Descripteur du module Test</description>

	<variables>
	  <!-- .... cf chapitre sur les variables .... -->
  </variables>

	<entry-points>
		<entry-point method-name="testPressureSync" name="TestPressureSync" where="compute-loop" property="none" />
		<entry-point method-name="dumpConnection" name="DumpConnection" where="compute-loop" property="none" />
	</entry-points>

	<options>
	</options>
</module>
```

La signification des attributs de l'élément **entry-point** est la suivante :
- **method-name** définit le nom de la méthode C++ correspondant au point d'entrée,
- **name** est le nom d'enregistrement du point d'entrée dans %Arcane,
- **property** donne le type de point d'entrée :
  - **none** : point d'entrée "traditionnel"
  - **auto-load-begin** : signifie que le module de ce point d'entrée 
    sera chargé automatiquement et que le point d'entrée sera appelé
    au début de la boucle en temps,
  - **auto-load-end** : signifie que le module de ce point d'entrée 
    sera chargé automatiquement et que le point d'entrée sera appelé
    à la fin de la boucle en temps
- **where** est l'endroit où le point d'entrée va être appelé. Les valeurs
  possibles sont :

<table>
<tr>
<td> **compute-loop**</td>
<td> appel du point d'entrée tant que l'on itère,</td>
</tr>
<tr>
<td> **init** </td>
<td>sert à initialiser les structures de données du module qui
ne sont pas conservées lors d'une protection. A ce stade de
l'initialisation, le jeu de données et le maillage ont déjà été lus.
L'initialisation sert également à vérifier certaines valeurs,
calculer des valeurs initiales...</td>
</tr>
<tr>
<td> **start-init** </td>
<td>sert à initialiser les variables et les valeurs uniquement lors du
démarrage du cas (t=0),</td>
</tr>
<tr>
<td> **continue-init** </td>
<td>sert à initialiser des structures spécifiques au mode reprise. A
priori, un module ne devrait pas avoir à faire d'opérations
spécifiques dans ce cas,</td>
</tr>
<tr>
<td> **build** </td>
<td>appel du point d'entrée avant l'initialisation ; le jeu de données
n'est pas encore lu. Ce point d'entrée sert généralement à construire
certains objets utiles au module mais est peu utilisé par les modules
numériques.</td>
</tr>
<tr>
<td> **on-mesh-changed** </td>
<td>sert à initialiser des variables et des valeurs lors d'un
changement de la structure de maillage (partitionnement, abandon de
mailles...). <strong>Attention</strong> : la taille des variables du
code définies sur des entités du maillage est automatiquement mise à
jour par %Arcane.</td>
</tr>
<tr>
<td> **restore** </td>
<td>sert à initialiser des structures spécifiques lors d'un retour arrière,</td>
</tr>
<tr>
<td> **exit** </td>
<td> sert, par exemple, à désallouer des structures de
données lors de la sortie du code : fin de simulation, arrêt avant
reprise...</td>
</tr>
</table>

Lors de la compilation du descripteur de module par %Arcane (avec **axl2cc** - cf précédemment), 
les points d'entrée sont enregistrés au sein de la base de données de l'architecture.

Il faut déclarer les points d'entrée au niveau de la classe du module (sinon une erreur
se produit à la compilation):

```cpp
class TestModule
{
  ...

 public:

   virtual void testPressureSync();
   virtual void dumpConnection();
   ...
};
```
	
## Construction {#arcanedoc_core_types_axl_entrypoint_build}

Les points d'entrée sont définis dans le fichier de définition
du module, dans notre cas \c TestModule.cc.

Pour exemple, voici le point d'entrée \c testPressureSync
appelé à chaque itération de la boucle de calcul. 
Ce point d'entrée effectue une
moyenne des pressions des mailles au cours du temps:

```cpp
using namespace Arcane;

VariableNodeReal m_node_pressure = ...;
VariableCellReal m_cell_pressure = ...;

void TestModule::
testPressureSync()
{
  m_global_deltat = options()->deltatInit();
  m_node_pressure.fill(0.0);

  // Ajoute à chaque noeud la pression de chaque maille auquel il appartient
  ENUMERATE_CELL(i,allCells()){
    Cell cell = *i;
    Real cell_pressure = m_pressure[i];
    for( NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode )
      m_node_pressure[inode] += pressure;
  }

  // Calcule la pression moyenne.
  ENUMERATE_NODE(i,allNodes()){
    Node node = *i;
    m_node_pressure[i] /= node.nbCell();
  }

  // Affecte à chaque maille la pression moyenne des noeuds qui la compose
  ENUMERATE_CELL(i,allCells()){
    Cell cell = *i;
    Integer nb_node = cell.nbNode();
    Real cell_pressure = 0.;
    for( NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode )
      cell_pressure += m_node_pressure[inode];
    cell_pressure /= nb_node;
    m_pressure[i] = cell_pressure;
  }

  // Synchronise la pression (pour une exécution parallèle)
  m_pressure.synchronize();

 // Calcule la pression minimale, maximale et moyenne sur l'ensemble des
 // domaines
 Real min_pressure = 1.0e10;
 Real max_pressure = 0.0;
 Real sum_pressure = 0.0;

 ENUMERATE_CELL(i,ownCells()){
   Real pressure = m_pressure[i];
   sum_pressure += pressure;
   if (pressure<min_pressure)
     min_pressure = pressure;
   if (pressure>max_pressure)
     max_pressure = pressure;
 }

 Real gmin = parallelMng()->reduce(Parallel::ReduceMin,min_pressure);
 Real gmax = parallelMng()->reduce(Parallel::ReduceMax,max_pressure);
 Real gsum = parallelMng()->reduce(Parallel::ReduceSum,sum_pressure);

 info() << "Local  Pressure: " << " Sum " << sum_pressure
        << " Min " << min_pressure << " Max " << max_pressure;
 info() << "Global Pressure: " << " Sum " << gsum
        << " Min " << gmin << " Max " << gmax;
}
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_variable
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions
</span>
</div>