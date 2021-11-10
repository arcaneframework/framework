Les types fondamentaux {#arcanedoc_core}
=======================

[TOC]

Il existe 4 types fondamentaux dans %Arcane, qui correspondent aux
notions de <em>Module</em>, <em>Variable</em>, 
<em>Point d'entrée</em> et <em>Service</em>, .

Pour une description sommaire de ces notions, se reporter à la
section \ref arcane_overview "vue d'ensemble d'ARCANE".

<ul>
<li>\ref arcanedoc_module "Module"</li>
<li>\ref arcanedoc_variable "Variable"</li>
<li>\ref arcanedoc_entrypoint "Point d'entrée"</li>
<li>\ref arcanedoc_service "Service"</li>
</ul>
  
Les modules {#arcanedoc_module}
==============

Un module est un ensemble de <em>points d'entrée</em> et 
de <em>variables</em>. Il peut posséder des options de configuration 
qui permettent à l'utilisateur de paramètrer le module via le jeu de 
données de la simulation.

Un module est représenté par une classe et un fichier XML appelé 
<em>descripteur de module</em>.

Le descripteur de module {#arcanedoc_module_desc}
-------------------------

Le descripteur de module est un fichier XML ayant 
l'extension ".axl".Il présente les caractéristiques du module :
- ses variables,
- ses points d'entrée,
- ses options de configuration.

~~~~~~~~~~~~~~~~{.xml}
<?xml version="1.0"?>
<module name="Test" version="1.0">
	<name lang="fr">Test</name>

	<description>Descripteur du module Test</description>

	<variables/>

	<entry-points/>

	<options/>
  </module>
~~~~~~~~~~~~~~~~
  
Par exemple, le fichier \c Test.axl ci-dessus présente le module 
nommé <em>Test</em> dont la classe de base, est \c BasicModule (cas général).
<strong>TODO: ajouter doc référence sur les autres attributs du module.</strong>

Ce module \c Test ne possède ni variable, ni point d'entrée, 
ni option de configuration. 

Les variables et les points d'entrée seront décrits dans les 
chapitres suivants.
Les options de configuration sont présentées dans le document 
\ref arcanedoc_caseoptions.

La classe représentant le module {#arcanedoc_module_class}
----------------------------------

Grâce à l'utilitaire \c axl2cc, le fichier \c Test.axl 
génère un fichier Test_axl.h. Ce fichier contient 
la classe \c ArcaneTestObject, classe de base du module de Test.

~~~~~~~~~~~~~~~~{.cpp}
class TestModule
: public ArcaneTestObject
{
 public:
  // Construit un module avec les paramètres spécifiés dans \a mb
  ModuleTest(const ModuleBuildInfo & mbi)
  : ArcaneTestObject(mbi) {}

  // Retourne le numéro de version du module
  virtual VersionInfo versionInfo() const { return VersionInfo(1, 0, 0); }
};
~~~~~~~~~~~~~~~~

L'exemple précédent montre que %Arcane impose que le constructeur du module 
prenne un object de type \c ModuleBuildInfo en paramètre pour le transmettre
à sa classe de base. %Arcane impose également la définition d'une méthode
\c versionInfo() qui retourne le numéro de version de votre module.

\note
Le fait de dériver de la classe ArcaneTestObject donne accés, entre autre,
aux traces %Arcane (cf \ref arcanedoc_traces) et aux méthodes suivantes :  
<table>
<tr><th>Méthode</th><th>Action</th></tr>
<tr><td>\c allCells() </td><td> retourne le groupe de toutes les mailles </td></tr>
<tr><td>\c allNodes() </td><td> retourne le groupe de tous les noeuds </td></tr>
<tr><td>\c allFaces() </td><td> retourne le groupe de toutes les faces </td></tr>
<tr><td>\c ownCells() </td><td> retourne le groupe des mailles propres au sous-domaine </td></tr>
<tr><td>\c ownNodes() </td><td> retourne le groupe des noeuds propres au
<tr><td>\c ownFaces() </td><td> retourne le groupe de toutes les faces propres au sous-domaine </td></tr>
</table>

Connexion du module à Arcane {#arcanedoc_module_connectarcane}
----------------------------

Une instance du module est contruite par l'architecture lors de l'exécution. 

L'utilisateur doit donc fournir une fonction pour créer une instance 
de la classe du module. %Arcane fournit une macro permettant de définir
une fonction générique de création. Cette macro doit être écrite dans 
le fichier source où est défini le module. Elle possède le prototype suivant :

~~~~~~~~~~~~~~~~{.cpp}
ARCANE_REGISTER_MODULE_TEST(TestModule);
~~~~~~~~~~~~~~~~

\c *TestModule* correspond au nom de la classe et *TEST* qui suit 
**ARCANE_REGISTER_MODULE_** permet de définir la fonction de création.

Les variables {#arcanedoc_variable}
==============

Une variable est une valeur manipulée par le code et gérée par
%Arcane. Par exemple le volume, la vitesse, sont des variables. Elle
sont caractérisées par un **nom**, un **type**, un **support**
et une **dimension**.

Les variables sont déclarées à l'intérieur d'un module, au sein 
du descripteur de module. 

Si deux modules utilisent des variables de même nom,
leurs valeurs seront implicitement partagées. C'est par ce moyen que les 
modules communiquent leurs informations.

Le type {#arcanedoc_variable_types}
-----------------	

Les **types** des variables sont :

| Nom C++      |  Type
|--------------|-----------------------------------------------------
| **Integer**  | entier signé (Arcane::Integer)
| **Int16**    | entier signé sur 16 bits (Arcane::Int16)
| **Int32**    | entier signé sur 32 bits (Arcane::Int32)
| **Int64**    | entier signé sur 64 bits (Arcane::Int64)
| **Byte**     | représente un caractère sur 8 bits (Arcane::Byte)
| **Real**     | réel IEEE 754 (Arcane::Real)
| **Real2**    | coordonnée 2D, vecteur de deux réels (Arcane::Real2)
| **Real3**    | coordonnée 3D, vecteur de trois réels (Arcane::Real3)
| **Real2x2**  | tenseur 2D, vecteur de quatre réels (Arcane::Real2x2)
| **Real3x3**  | tenseur 3D, vecteur de neufs réels (Arcane::Real3x3)
| **String**   | chaîne de caractères unicode (Arcane::String)

Par défaut, les entiers sont stockés sur 4 octets mais il est possible 
de passer sur 8 octets en compilant avec la macro \c ARCANE_64BIT.
Les flottants (*Real*, *Real2*, *Real2x2*, *Real3*, *Real3x3*) sont des réels double précisions (stockés
sur 8 octets).

Le support {#arcanedoc_variable_support}
-----------------------

Le **support** correspond à l'entité qui porte la variable,
sur laquelle la variable est définie. Ces variables qui s'appliquent sur des éléments du maillage
sont appelées des **grandeurs**.

| Nom C++      | Support
|--------------|-----------------------------------------------------
| (vide)       | variable définie globalement (ex : pas de temps)
| **Node**     | noeud du maillage (Arcane::Node)
| **Face**     | face du maillage (Arcane::Face)
| **Cell**     | maille du maillage (Arcane::Cell)
| **Particle** |  particule du maillage (Arcane::Particle)

La dimension {#arcanedoc_variable_dim}
--------------

La **dimension** peut être:

| Nom C++     | Dimension
|-------------|------------
| **Scalar**  | scalaire
| **Array**   | tableau 1D
| **Array2**  | tableau 2D

Classe C++ {#arcanedoc_variable_cppclass}
-------------

Il est aisé d'obtenir la classe C++ correspondant à un type, un support 
et à une dimension donnés. Le nom de la classe est construit de la 
manière suivante :

**Variable** + **support** + **dimension** + **type**

Par exemple, pour une variable représentant un tableau d'entiers
**VariableArrayInteger**, pour une variable représentant un 
réel **VariableScalarReal**.

Quand une variable scalaire est définie sur une entité du maillage
le support (*Scalar*) n'est pas précisé. Par exemple, pour une variable 
représentant un réel aux mailles **VariableCellReal**.

Tous les combinaisons sont possibles aux exceptions suivantes:
- les variables de type *chaîne de caractères* qui n'existent que pour les genres
  scalaires et tableaux mais pas sur les éléments du maillage (pour
  des raisons de performances).
- Les variables de dimension 2 ne peuvent pas avoir de support (il
  n'est pas possible d'avoir des variables 2D aux mailles par exemple.
	
Le tableau suivant donne quelques exemples de variables:

Nom C++                          | Description
---------------------------------|------------
Arcane::VariableScalarReal       | un réel
Arcane::VariableScalarInteger    | un entier
Arcane::VariableArrayInteger     | Tableau d'entiers
Arcane::VariableArrayReal3       | Tableau de coordonnées 3D
Arcane::VariableNodeReal2        | Coordonnée 2D aux noeuds
Arcane::VariableFaceReal         | Réel aux faces
Arcane::VariableFaceReal3        | Coordonnée 3D aux faces
Arcane::VariableFaceArrayInteger | Tableau d'entiers aux faces
Arcane::VariableCellArrayReal    | Tableau de réels aux mailles
Arcane::VariableCellArrayReal3   | Tableau de coordonnées 3D aux mailles
Arcane::VariableCellArrayReal2x2 |Tableau de tenseurs 2D aux mailles

Déclaration {#arcanedoc_variable_declare}
-------------

La déclaration des variables se fait par l'intermédiaire du 
descripteur un module.

Par exemple, on déclare dans le module \c Test une
variable de type réel aux mailles appelée \c Pressure et
une variable de type réel aux noeuds appelée \c NodePressure.

~~~~~~~~~~~~~~~~{.xml}
<module name="Test" version="1.0">
	<name lang="fr">Test</name>

	<description>Descripteur du module Test</description>

	<variables>
 		<variable
			field-name="pressure"
			name="Pressure"
			data-type="real"
			item-kind="cell"
			dim="0"
			dump="true"
			need-sync="true" />
		<variable
			field-name="node_pressure"
			name="NodePressure"
			data-type="real"
			item-kind="node"
			dim="0"
			dump="true"
			need-sync="true" />
   </variables>

	<entry-points/>

	<options/>
</module>
~~~~~~~~~~~~~~~~

Les attributs suivants sont disponibles:

<table>
<tr>
<th>Attribut</th>
<th>Description</th>
</tr>
<tr>
<td> **name** </td>
<td>nom de la variable. Par convention, elle commence par une
majuscule. Les caractères valides sont les caractères alphabétiques
**[a-zA-Z]**, les chiffres (sauf au début du nom) et le caractère
souligné. Les variables dont le nom commence par *Global* et *%Arcane*
sont réservées pour %Arcane.
</td>
</tr>
<tr>
<td> **field-name** </td>
<td>nom information du champ de la variable dans la classe générée. Ce nom doit
être un nom valide en C++. Par convention, si la variable s'appelle
 *NodePressure* par exemple, le champ sera **node_pressure**. Dans la
 classe générée, le nom spécifié sera préfixé de **m_** (le préfixage \c m_ correspond aux 
  \ref arcanedoc_codingrules "règles de codage" dans %Arcane pour les attributs de classe).
</td>
</tr>
<tr>
<td> **item-kind** </td>
<td>support de la variable. Les valeurs possibles sont  \c node, \c
face, \c cell ou \c none. La valeur \c none indique qu'il ne s'agit
pas d'une variable du maillage. A noter qu'il n'est pas (encore)
possible de déclarer dans le fichier axl les variables ayant comme
support des particules (Particle) ou liens (Link).
</td>
</tr>
<tr>
<td> **data-type** </td>
<td>type de données de la variable. Il s'agit au choix de *integer*, *int16*,
 *int32*, *int64*, *real*, *string*, *real2*, *real3*, *real2x2*,
 *real3x3*. Le type *string8 n'est pas utilisable pour les variables
 du maillage (celles ayant un support).
</td>
</tr>
<tr>
<td> **dim** </td>
<td>La dimension de la variables. Les valeurs possibles sont *0* pour
une variable scalaire, *1* pour une variable tableau à une dimension et
*2* pour un tableau à deux dimensions. Les tableaux à deux dimensions
ne sont pas supportés pour les variables du maillage.
</td>
</tr>
<tr>
<td> **dump** </td>
<td>permet de choisir si les valeurs d'une variable sont sauvegardées
lors d'un arrêt du code. Dans ce cas, les valeurs sauvegardées seront,
bien entendu, relues lors d'une reprise d'exécution. Certaines
variables recalculées n'ont pas besoin d'être sauvegardées ; dans ce
cas l'attribut *dump* vaut *fals*e. C'est le cas lorsque la valeur
d'une variable n'est pas utile d'une itération à l'autre.
</td>
</tr>
<tr>
<td> **need-sync** </td>
<td>indique si la variable doit être synchronisé entre
sous-domaines. Il s'agit juste d'une indication qui peut être utilisée
lors de vérifications.
</td>
</tr>
</table>

Lors de la compilation du descripteur de module par %Arcane (avec \c axl2cc - cf précédemment), 
les variables sont enregistrées au sein de la base de données de l'architecture.

Utilisation {#arcanedoc_variable_use}
-----------------------  

La manière d'utiliser une variable est identique quel que soit son
type et ne dépend que de son genre.

**Variables scalaires**
	
Les variables scalaires sont utilisées par l'intermédiaire de la classe template
Arcane::VariableRefScalarT.

Il n'y a que deux possibilités d'utilisation :

- "lire la valeur" : cela se fait par
  l'opérateur() (VariableRefScalarT::operator()()). Cet opérateur retourne une
  référence constante sur la valeur stockée dans la variable. Il peut être utilisé
  partout ou l'on veut utiliser une valeur du type de la variable.
- "changer la valeur" : cela se fait par l'opérateur = (Arcane::VariableRefScalarT::operator=())

Par exemple, avec la variable \c m_time de type \c VariableScalarReal :

~~~~~~~~~~~~~~~~{.cpp}
m_time = 5.;         // affecte la valeur 5. à la variable m_time
double z = m_time(); // récupère la valeur de la variable et l'affecte à z.
cout << m_time();    // imprime la valeur de m_time
~~~~~~~~~~~~~~~~

L'important est de ne pas oublier les parenthèses lorsqu'on veut
accéder à la valeur de la variable.

**Variables tableaux**
	
Les variables tableaux sont utilisées par l'intermédiaire de la classe template
*Arcane::VariableRefArrayT*.

Leur fonctionnement est assez similaire à la classe \c vector
de la STL. Le dimensionnement du tableau se fait par la méthode
Arcane::VariableRefArrayT::resize() et chaque élément du tableau peut être accédé par l'opérateur
Arcane::VariableRefArrayT::operator[]() qui retourne une référence sur le type des éléments du
tableau.

Par exemple, avec la variable *m_times* de type *VariableArrayReal*:

~~~~~~~~~~~~~~~~{.cpp}
Arcane::VariableArrayReal m_times = ...;
m_times.resize(5);         // redimensionne le tableau pour contenir 5 éléments
m_times[3] = 2.0;          // affecte la valeur 2.0 au 4ème élément du tableau
cout << m_times[0];        // imprime la valeur du premier élément
~~~~~~~~~~~~~~~~

**Variables scalaires sur le maillage**
	
Il s'agit des variables sur les éléments du maillages (noeuds,
faces ou mailles) avec une valeur par élément. Ces variables sont
définies par la classe template Arcane::MeshVariableScalarRefT.

Leur fonctionnement est assez similaire à celui d'un tableau C
standard. On utilise l'opérateur Arcane::VariableRefArrayT::operator[]()
pour récupérer une référence sur le type de la variable pour un
élément du maillage donné. Cet opérateur est surchargé pour prendre
en argument un itérateur sur un élément du maillage.

Les grandeurs se déclarent et s'utilisent de manière similaire
quels que soient le type d'élément du maillage. Elles sont dimensionnées
automatiquement lors de l'initialisation au nombre d'éléments du
maillage du genre de la variable.

Par exemple, avec la variable *m_volume* de type *Arcane::VariableCellReal*:

~~~~~~~~~~~~~~~~{.cpp}
Arcane::VariableCellReal m_volume = ...;
ENUMERATE_CELL(i,allCells()) {
  m_volume[i] = 2.0;     // Affecte la valeur 2.0 au volume de la maille courante
  cout << m_volume[i];   // Imprime le volume de la maille courante

  // il est possible de faire les mêmes opérations avec la maille
  // ATTENTION c'est moins performant
  Cell cell = *i;         // Déclare une référence à une maille.
  m_volume[cell] = 2.0;   // Affecte la valeur 2.0 au volume de la maille 'cell'
  cout << m_volume[cell]; // Imprime le volume de la maille 'cell'
}
~~~~~~~~~~~~~~~~

**Variables tableaux sur le maillage**

Il s'agit des variables sur les éléments du maillage (noeuds,
faces ou mailles) avec un tableau de valeurs par éléments. Ces variables sont
définies par la classe template Arcane::MeshVariableArrayRefT.

Le fonctionnement de ces variables est identique à celui des
variables scalaires sur le maillage mais l'opérateur
Arcane::MeshVariableArrayRefT::operator[]() retourne un tableau de valeurs
du type de la variable.

Il est possible de changer le nombre d'éléments de la deuxième
dimension de ce tableau par la méthode Arcane::MeshVariableArrayRefT::resize().

Par exemple, avec la variable *m_temperature* de type *Arcane::VariableCellArrayReal*:

~~~~~~~~~~~~~~~~{.cpp}
Arcane::VariableCellArrayReal m_temperature = ...;
m_temperature.resize(3); // Chaque maille aura 3 valeurs de temperature
ENUMERATE_CELL(i,allCells()) {
  m_volume[i][0] = 2.0;        // Affecte la valeur 2.0 à la première temperature de la maille courante
  cout << m_volume[i][1];      // Imprime la 2ème température de la maille courante

  // il est possible de faire les mêmes opérations avec la maille
  // ATTENTION c'est moins performant
  Cell cell = *i;            // Déclare une référence à une maille.
  m_volume[cell][0] = 2.0;   // Affecte la valeur 2.0 à la première temperature de la maille 'cell'
  cout << m_volume[cell][1]; // Imprime la 2ème température de la maille 'cell'
}
~~~~~~~~~~~~~~~~

Les points d'entrée {#arcanedoc_entrypoint}
==================

Un point d'entrée est une méthode d'un module qui est référencée par %Arcane
et qui sert d'interface pour le module avec la boucle en temps. Un point d'entrée est décrit par
la classe interface \c IEntryPoint. Un point d'entrée est une méthode dont la signature est la
suivante, <b>T</b> étant le type de la classe du module:

~~~~~~~~~~~~~~~~{.cpp}
void T::func();
~~~~~~~~~~~~~~~~

Un point d'entrée est caractérisé par:
- un nom
- une méthode de la classe associée.
- l'endroit où il peut être appelé (initialisation, boucle de calcul, ...).
Par défaut, un point d'entrée est appelé dans la boucle de calcul.

Les points d'entrée sont déclarés dans le descripteur de module. Par exemple, pour le 
module \c Test, on déclare 2 points d'entrée pouvant ensuite être appelés dans 
la boucle en temps par les noms <b>DumpConnection</b> ou <b>TestPressureSync</b> :

~~~~~~~~~~~~~~~~{.xml}
<module name="Test" version="1.0">
	<name lang="fr">Test</name>

	<description>Descripteur du module Test</description>

	<variables>
	        <!-- .... cf chapitre sur les variables .... -->
       </variables>

	<entry-points>
		<entry-point method-name="testPressureSync" name="TestPressureSync" where="compute-loop" property="none" />
		<entry-point method-name="dumpConnection" name="DumpConnection" where="compute-loop" property="none" />
	<entry-points>

	<options/>
</module>
~~~~~~~~~~~~~~~~

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

~~~~~~~~~~~~~~~~{.cpp}
class TestModule
{
  ...

 public:

   virtual void testPressureSync();
   virtual void dumpConnection();
   ...
};
~~~~~~~~~~~~~~~~
	
Construction {#arcanedoc_module_build}
-------------------

Les points d'entrée sont définis dans le fichier de définition
du module, dans notre cas \c TestModule.cc.

Pour exemple, voici le point d'entrée \c testPressureSync
appelé à chaque itération de la boucle de calcul. 
Ce point d'entrée effectue une
moyenne des pressions des mailles au cours du temps:

~~~~~~~~~~~~~~~~{.cpp}
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
~~~~~~~~~~~~~~~~

Les services {#arcanedoc_service}
====================

Un service présente les mêmes caractéristiques qu'un module,
sauf qu'il ne possède pas de point d'entrée. Tout comme le module, 
il peut donc posséder des variables et des options de configuration.

Le service est représenté par une classe et 
un fichier XML appelé <em>descripteur de service</em>.

Les services sont généralement utilisés :
- pour capitaliser du code entre plusieurs modules. Par exemple, 
  on peut créer un service de schéma numérique utilisé par plusieurs 
  modules numériques comme la thermique et l'hydrodynamique.
- pour paramétrer un module avec plusieurs algorithmes. Par exemple,
  on peut faire un service chargé d'appliquer une équation d'état
  pour le module d'hydrodynamique. On définit alors 2 implémentations
  "gaz parfait" et "stiffened gaz" et on paramètre le module, via
  son jeu de données, avec l'une ou l'autre des implémentations.

Du point de vue de la conception, cela implique :
- de déclarer une interface qui va être le contrat du service.
  Par exemple, voici l'interface d'un service pour la résolution d'une équation
  d'état sur un groupe de mailles passé en argument :
~~~~~~~~~~~~~~~~{.cpp}
class IEquationOfState
{
 public:
  virtual void applyEOS(const Arcane::CellGroup& group) =0;
};
~~~~~~~~~~~~~~~~
- de créer une ou plusieurs implémentations de cette interface.

Une instance de service est créé lors de la lecture du jeu de données
du module qui utilise ce service. Les méthodes du service peuvent alors être 
appelées directement depuis le module.

\note
Les options de configuration du module, notamment celle permettant de
référencer un service, sont présentées dans le document 
\ref arcanedoc_caseoptions.
  
Le descripteur de service {#arcanedoc_service_desc}
---------------------------------------------------

Comme pour le module, le descripteur de service est un fichier XML ayant 
l'extension ".axl". Il présente les caractéristiques du service:
- la où les interfaces qu'il implémente,
- ses options de configuration,
- ses variables.

Contrairement au module, un service n'a pas de points d'entrée.

L'exemple suivant définit un descripteur de service pour la résolution
d'une équation d'état de type "gaz parfait" :

~~~~~~~~~~~~~~~~{.xml}
<?xml version="1.0"?>
<service name="PerfectGasEOS" version="1.0">
  <description>Jeu de données du service PerfectGasEOS</description>
  <interface name="IEquationOfState" />

<options> ... Liste des options ... /options>
  <variables> ... Liste des variables ... </variables>
</service>
~~~~~~~~~~~~~~~~
  
L'exemple suivant définit un descripteur de service utilisé pour la résolution
d'une équation d'état de type "stiffened gaz" :

~~~~~~~~~~~~~~~~{.xml}
<?xml version="1.0"?>
<service name="StiffenedGasEOS" version="1.0" type="caseoption">
  <description>Jeu de données du service StiffenedGasEOS</description>
  <interface name="IEquationOfState" />

	<options> ... Liste des options ... /options>
  <variables> ... Liste des variables ... </variables>
</service>
~~~~~~~~~~~~~~~~

L'attribut *type* valant *caseoption* indique qu'il s'agit d'un
service pouvant être référencé dans un jeu de données.

Il est aussi possible de spécifier un attribut *singleton* ayant
comme valeur un booléen indiquant si le service peut être singleton.

La classe représentant le service {#arcanedoc_module_class}
----------------------------------

Comme pour le module, la compilation des fichiers \c PerfectGas.axl 
et \c StiffenedGas.axl avec l'utilitaire \c axl2cc génère respectiment
les fichiers PerfectGas_axl.h et StiffenedGas_axl.h contenant les classes 
\c ArcanePerfectGasObject et \c ArcaneStiffenedGasObject, classes de 
base des services.

Voici les classes pour les services définis précédemment dans les
descripteurs :

~~~~~~~~~~~~~~~~{.cpp}
class PerfectGasEOSService 
: public ArcanePerfectGasEOSObject
{
 public:
   explicit PerfectGasEOSService(const Arcane::ServiceBuildInfo& sbi)
	 : ArcanePerfectGasEOSObject(sbi) {}

 public:
   void applyEOS(const Arcane::CellGroup& group) override
   { // ... corps de la méthode }
};
~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~{.cpp}
class StiffenedGasEOSService 
: public ArcaneStiffenedGasEOSObject
{
 public:
   explicit StiffenedGasEOSService(const Arcane::ServiceBuildInfo& sbi)
	 : ArcaneStiffenedGasEOSObject(sbi) {}
	
 public:
   void applyEOS(const Arcane::CellGroup& group) override
   { // ... corps de la méthode }
};
~~~~~~~~~~~~~~~~

L'exemple précédent montre que %Arcane impose que le constructeur d'un service 
prenne un object de type \c ServiceBuildInfo en paramètre pour le transmettre
à sa classe de base. On peut également constater que le service hérite
de l'interface définissant le contrat du service.

Connexion du service à Arcane {#arcanedoc_service_connectarcane}
----------------------------------

Une instance du service est contruite par l'architecture lorsque un module
référence le service dans son jeu de données. 

L'utilisateur doit donc fournir une fonction pour créer une instance 
de la classe du service. %Arcane fournit une macro permettant de définir
une fonction générique de création. Cette macro doit être écrite dans 
le fichier source où est défini le service.

Voici cette macro pour les exemples précédents :

~~~~~~~~~~~~~~~~{.cpp}
ARCANE_REGISTER_SERVICE_PERFECTGASEOS(PerfectGas, PerfectGasEOSService);
ARCANE_REGISTER_SERVICE_STIFFENEDGASEOS(StiffenedGas, StiffenedGasEOSService);
~~~~~~~~~~~~~~~~

*PerfectGas* et *StiffenedGas* correspondent aux noms 
d'enregistrement dans %Arcane et donc aux noms par 
lesquels les services seront référencés dans le jeu de données des modules.
*PerfectGasEOSService* et *StiffenedGasEOSService* correspondent aux noms 
des classes C++.et les noms qui suivent **ARCANE_REGISTER_SERVICE_**
permettent de définir la fonction de création.

Il est toutefois possible d'enregistrer un service même si celui-ci ne possède pas
de fichier axl. Cela se fait par la macro ARCANE_REGISTER_SERVICE(). Par exemple,
pour enregistrer la class *MyClass* comme un service de sous-domaine de nom 'Toto', qui
implémente l'interface 'IToto', on écrira:

~~~~~~~~~~~~~~~~{.cpp}
ARCANE_REGISTER_SERVICE(MyClass,
                        ServiceProperty("Toto",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IToto));
~~~~~~~~~~~~~~~~

