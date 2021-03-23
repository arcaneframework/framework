Configuration des options jeu de données (fichier AXL){#arcanedoc_caseoptions}
======================

[TOC]

Introduction {#arcanedoc_caseoptions_intro}
=============

Ce chapître décrit les options possibles pour le fichier *axl*. Ces options
s'appliquent de manière identiques aux modules et aux services. Afin
d'éviter des répétitions inutiles, on utilisera le terme module
seulement, en sachant que cela s'applique aussi aux services.

Chaque module possède des options qui peuvent être spécifiées
par l'utilisateur lors du lancement d'une exécution. Ces options sont en
général dictées par le *jeu de données* que fournit
l'utilisateur pour lancer son cas. Le document \ref arcanedoc_module 
montre que chaque module possède un fichier de configuration nommé
*descripteur de module* composé de 3 parties : 
les variables, les points d'entrée et les options de configuration.
Le présent document s'intéresse à la la partie concernant les
options de configuration qui vont permettre de définir la grammaire
du jeu de données du module.

Le descripteur de module est un fichier au format XML. Ce fichier 
est utilisé par %Arcane pour générer des classes C++. Une de ces
classes se charge de la lecture des informations dans le jeu
de données.

Par convention, pour un module appelé *Test*, le descripteur
de module s'appelle *Test.axl*. Ce fichier `axl` permet de générer un fichier
*Test_axl.h*. Ce fichier sera inclu par la classe implémentant le module.

Dans le jeu de données, les options d'un module sont fournies
dans l'élément de nom `<options>`. Par exemple, les options
du module *Test* sont :

~~~~~~~~~~~~~~~~{.xml}
<module name="Test" version="1.0">
  <name lang="fr">Test</name>
  <description>Module Test</description>
  <options>
    <!-- contient les options du module Test -->
    ...
  </options>
</module>
~~~~~~~~~~~~~~~~

Structure du fichier {#arcanedoc_caseoptions_struct}
=====================

Le descripteur de module est au format XML. Nous allons nous
intéresser à la partie configuration des options contenue dans 
l'élément \c options de ce fichier. En voici un exemple :

~~~~~~~~~~~~~~~~{.xml}
<options>
  <simple name = "simple-real" type = "real">
    <name lang='fr'>reel-simple</name>
    <description>Réel simple</description>
  </simple>
</options>
~~~~~~~~~~~~~~~~

Cet exemple définit une option de configuration appelée
*simple-real*. Cette option est une variable simple de type
`real` sans valeur par défaut.

La structure de tout élément de configuration des options d'un
module est similaire à celle-ci. Toutes les options possibles doivent
apparaître dans des éléments fils de \c options.

Les différentes possibilités sont les suivantes :
- les options simples, de type `real`, `bool`,
  `integer` ou `string`.
- les options énumérées, qui doivent correspondre à un type
  `enum` du C++.
- les options de types dit étendus. Il s'agit de types créés
  par l'utilisateur (classes, structures...).  Cela comprend 
  par exemple les groupes d'entités du maillage.
- les options complexes, qui sont composées elles-mêmes d'options.
  Les options complexes peuvent s'imbriquer.
- les options services, qui permettent de référencer un service (voir le document \ref arcanedoc_service).

Attributs et propriétés communs à toutes les options {#arcanedoc_caseoptions_struct_common}
-----------------------------------------------------

Quelle que soit l'option, l'élément la définissant doit comporter les
deux attributs suivants:

<table>
<tr>
<th>attribut</th>
<th>occurence</th>
<th>description</th>
<tr>
<td>`name`</td>
<td>requis</td>
<td>nom de l'option. Il doit être composé uniquement de caractères
alphabétiques en minuscules avec le '-' comme séparateur. Ce nom est
utilisé pour générer dans la classe C++ un champ de même nom où les
'-' sont remplacés par des majuscules. Par exemple, une option nommée
`max-epsilon` sera récupérée dans le code par la méthode `maxEpsilon()`,
</td>
</tr>
<tr>
<td>`type`</td>
<td>requis</td>
<td>type de l'option. La valeur et la signification de cet attribut dépendent du type de l'option.</td>
</tr>
<tr>
<td>`default`</td>
<td>optionnel</td>
<td>valeur par défaut de l'option. Si l'option n'est pas présente
dans le jeu de données, %Arcane fera comme si la valeur de cet attribut
avait été entrée par l'utilisateur. L'option vaudra alors la valeur
fournie dans l'attribut \c default. Si cet attribut n'est pas présent,
l'option n'a pas de valeur par défaut et doit toujours être présente
dans le jeu de données.</td>
</tr>
<tr>
<td>`minOccurs`</td>
<td>optionnel</td>
<td>entier qui spécifie le nombre minimum d'occurences
possible pour l'élément. Si cette valeur vaut zéro, l'option peut être
omise même si l'attribut `default` est absent. Si cet attribut
est absent, le nombre minimum d'occurence est 1.</td>
</tr>
<tr>
<td>`maxOccurs`</td>
<td>optionnel</td>
<td>entier qui spécifie le nombre maximum d'occurences
possible pour l'élément. Cette valeur doit être supérieur ou égal à `minOccurs`. La valeur spéciale `unbounded` signifie que le
nombre maximum d'occurences n'est pas limité. Si cet attribut est
absent, le nombre maximum d'occurence est 1.</td>
</tr>
</table>
  
Pour chaque option, il est possible d'ajouter les éléments fils
suivants :

<table>
<tr>
<th>élément</th>
<th>occurence</th>
<th>description</th>
</tr>
<tr>
<td>`description`</td>
<td>optionnel</td>
<td> qui est utilisé pour décrire l'utilisation de
      l'option. Cette description peut utiliser des éléments HTML. Le contenu
      de cet élément est repris par %Arcane pour la génération de la documentation
      du jeu de données.
</td>
</tr>
<tr>
<td>`userclass`</td>
<td>optionnel</td>
<td> indique la classe d'appartenance de l'option.
      Cette classe spécifie une catégorie d'utilisateur ce qui permet par
      exemple de restreindre certaines options à une certaine catégorie. Par
      défaut, si cet élément est absent, l'option n'est utilisable que
      pour la classe utilisateur. Il est possible de spécifier plusieurs
      fois cet élément avec une catégorie différente à chaque fois. Dans ce
      cas, l'option appartient à toutes les catégories spécifiées.
</td>
</tr>
<tr>
<td>`defaultvalue`</td>
<td>0..infini</td>
<td> permet d'indiquer une valeur par défaut pour une catégorie
donnée. Par exemple:
~~~~~~~~~~~~~~~~{.xml}
<simple name="simple-real" type="real">
  <defaultvalue category="Code1">2.0</defaultvalue>
  <defaultvalue category="Code2">3.0</defaultvalue>
</simple>
~~~~~~~~~~~~~~~~

Dans l'exemple précédent, si la catégorie est 'Code1' alors la
valeur par défaut sera '2.0'. Il est possible de spécifier autant de
catégorie qu'on le souhaite. La catégorie utilisée lors de
l'exécution est positionnée via la méthode
Arcane::ICaseDocument::setDefaultCategory().
</td>
</tr>
<tr>
<td>`name`</td>
<td>0..infini</td>
<td>
permet d'indiquer une traduction pour le nom de
l'option. La valeur de cet élément est le nom traduit pour l'option
et correspondant au langage spécifié par l'attribut <tt>lang</tt>.
Par exemple :
~~~~~~~~~~~~~~~~{.xml}
<simple name="simple-real" type="real">
  <name lang='fr'>reel-simple</name>
</simple>
~~~~~~~~~~~~~~~~

indique que l'option 'simple-real' s'appelle en francais 'reel-simple'.
Plusieurs éléments <tt>name</tt> sont possibles, chacun spécifiant une
traduction. Le jeu de données devra être fourni dans la langue par défaut,
le français dans notre cas. Si aucune traduction n'est donnée, c'est
la valeur de l'attribut \c name qui est utilisée.
</td>
</tr>
</table>

Les options simples {#arcanedoc_caseoptions_struct_simple}
--------------------

Elles sont décrites par l'élément <tt>simple</tt> :

~~~~~~~~~~~~~~~~{.xml}
<simple name="simple-real" type="real">
  <description>Réel simple</description>
</simple>
~~~~~~~~~~~~~~~~

L'attribut `type` peut prendre une des valeurs suivantes :
- `real` pour les réels. Ils correspondent au type Arccore::Real.
- `integer` pour les entier. Ils correspondent au type Arccore::Integer.
- `int32` pour les entier. Ils correspondent au type Arccore::Int32.
- `int64` pour les entier. Ils correspondent au type Arccore::Int64.
- `bool` pour les booléens. Ils correspondent au type `bool` du C++.
- `string` pour les chaînes de caractères. Ils correspondent à la classe Arccore::String.

Pour l'exemple précédent, le jeu de données contient, par exemple, la ligne :

~~~~~~~~~~~~~~~~{.xml}
<simple-real>3.4</simple-real>
~~~~~~~~~~~~~~~~

Les options énumérées {#arcanedoc_caseoptions_struct_enum}
---------------------

Elles sont décrites par l'élément <tt>enumeration</tt> :

~~~~~~~~~~~~~~~~{.xml}
<enumeration name="boundary-condition" type="eBoundaryCondition" default="X">
  <description>Type de condition au limites</description>
  <enumvalue name="X" genvalue="VelocityX" />
  <enumvalue name="Y" genvalue="VelocityY"  />
  <enumvalue name="Z" genvalue="VelocityZ"  />
</enumeration>
~~~~~~~~~~~~~~~~

L'attribut \c type doit indiquer le nom de
l'énumération C++ correspondante (ici \c eBoundaryCondition). Si
l'énumération appartient à un espace de nom C++ (\c namespace), il
est possible de le spécifier directement (par exemple \c MonCode::eBoundaryCondition).
	
Ce type d'option comporte une liste d'éléments fils
<tt>enumvalue</tt> qui référence chaque valeur
de l'énumération. Cet élément comporte deux attributs :
- `name` indique à quelle valeur de l'utilisateur va
  correspondre cette énumération. Ce sont ces valeurs que
  l'utilisateur va entrer.
- `genvalue` indique le nom de l'énumération correspondante
  dans le langage C++.

Dans l'exemple précédent, l'option correspond à l'énumération C++
suivante :

~~~~~~~~~~~~~~~~{.cpp}
enum eBoundaryCondition { VelocityX, VelocityY, VelocityZ };
~~~~~~~~~~~~~~~~

La définition de l'énumération doit être effective avant d'inclure
le fichier *Test_axl.h*. Par exemple, si l'énumération
précédente est définie dans un fichier *BoundaryCondition.h*,
il faudra inclure les fichiers dans l'ordre suivant :
~~~~~~~~~~~~~~~~{.cpp}
#include "BoundaryCondition.h"
#include "Test_axl.h"
~~~~~~~~~~~~~~~~
  
Si dans le jeu de données, on a la ligne :

~~~~~~~~~~~~~~~~{.xml}
<boundary-condition>X</boundary-condition>
~~~~~~~~~~~~~~~~
 
alors l'option associée dans la classe C++ générée par le fichier aura
la valeur <tt>VelocityX</tt>.

Les options étendues {#arcanedoc_caseoptions_struct_extended}
---------------------

L'interface d'utilisation de ce type d'option est en cours de définition.
Pour l'instant, ce type d'option ne doit être utilisé qu'avec les 
groupes d'entités de maillage définis par %Arcane.

  Les options de type étendues sont décrites par l'élément <tt>extended</tt> :

~~~~~~~~~~~~~~~~{.xml}
<extended name="surface" type="Arcane::FaceGroup">
   <description>
     Surface sur laquelle s'applique la condition aux limites
   </description>
</extended>
~~~~~~~~~~~~~~~~

L'attribut <b>type</b> peut prendre une des valeurs suivantes :
  
- *Arcane::NodeGroup* pour un groupe de noeuds
- *Arcane::FaceGroup* pour un groupe de faces
- *Arcane::CellGroup* pour un groupe de mailles

Si une option de ce type existe, sa valeur est validée après lecture
du maillage et doit correspondre à un groupe existant et du bon type.

Pour le développeur, une option de type \c extended génère un objet du type
<tt>Arcane::NodeGroup</tt>, <tt>Arcane::FaceGroup</tt> 
ou <tt>Arcane::CellGroup</tt> qui s'utilise donc comme un groupe. Par exemple,
avec l'option précédente de nom <tt>surface</tt>, la méthode
<tt>surface()</tt> de la classe gérant le jeu de donnée retourne un
groupe de faces de type <tt>Arcane::FaceGroup</tt> et la ligne suivante est valide :

~~~~~~~~~~~~~~~~{.cpp}
ENUMERATE_FACE(i,surface())
~~~~~~~~~~~~~~~~

Pour l'exemple précédent, si XMIN est un surface définie sur le maillage,
le jeu de données contient, par exemple, la ligne :

~~~~~~~~~~~~~~~~{.xml}
<surface>XMIN</surface>
~~~~~~~~~~~~~~~~

Les options complexes {#arcanedoc_caseoptions_struct_complex}
----------------------

Une option complexe est composée de plusieurs autres options, y compris
d'autres options complexes. L'option complexe est décrite par l'élément <tt>complex</tt>.

L'exemple suivant définit une option <tt>pair-of-int</tt> composée
de deux options simples <tt>first</tt> et <tt>second</tt> :

~~~~~~~~~~~~~~~~{.xml}
<complex name="pair-of-int" type="PairOfInt">
  <simple name="first" type="integer">
  <simple name="second" type="integer">
</complex>
~~~~~~~~~~~~~~~~

Le jeu de données correspondant est :

~~~~~~~~~~~~~~~~{.xml}
<pair-of-int>
  <first>5</first>
  <second>6</second>
</pair-of-int>
~~~~~~~~~~~~~~~~

Les options de type service {#arcanedoc_caseoptions_struct_service}
----------------------------

La notion de service %Arcane est expliqué dans la section \ref arcanedoc_service.
Rappelons simplement qu'un service est un composant externe utilisé par un module
et donc nécessaire au fonctionnement du module. Pour que le module puisse
fonctionner, son jeu de données doit référencer le service nécessaire grâce à
l'élément <tt>service-instance</tt>.

L'exemple suivant définit un service nommé *eos-model* 
de type \c IEquationOfState :

~~~~~~~~~~~~~~~~{.xml}
<service-instance name="eos-model" type="IEquationOfState">
  <description>Service d'equation d'état</description>
</service-instance>
~~~~~~~~~~~~~~~~

La classe \c IEquationOfState correspond à l'interface définie pour le service.
Si la classe appartient à un espace de nom C++ (\c namespace), il est
possible de le spécifier directement (par exemple \c
MonCode::IEquationOfState).

Le jeu de données correspondant est:

~~~~~~~~~~~~~~~~{.xml}
<eos-model name="StiffenedGas">
  <limit-tension>0.01</limit-tension>
</eos-model>
~~~~~~~~~~~~~~~~

Le code pour lire l'option est:
  
~~~~~~~~~~~~~~~~{.cpp}
IEquationOfState* eos = options()->eosModel();
~~~~~~~~~~~~~~~~
  
Les services peuvent avoir les attributs suivants:
<table>
<tr>
<th>Attribut</th>
<th>Type</th>
<th>Description</th>
</tr>
<tr>
<td>allow-null</td>
<td>bool</td>
<td>si \c false (le défaut), le service spécifié dans le jeu de
données doit exister. Si ce n'est pas le cas, une erreur se produira
lors de la lecture. Si cet attribut vaut \c true, et que le service
spécifié n'existe pas, l'option correspondante sera nulle et il
faudra tester sa valeur. Par exemple:
~~~~~~~~~~~~~~~~{.cpp}
IEquationOfState* eos = options()->eosModel();
if (!eos)
  info() << "Service spécifié indisponible";
~~~~~~~~~~~~~~~~
</td>
</tr>
<tr>
<td>optional</td>
<td>bool</td>
<td>si \c false (le défaut), le service est toujours créé et si
l'élément correspondant n'existe pas dans le jeu de données, le
service avec le nom par défaut sera créé. Si cet attribut vaut \c
true et que l'élément n'est pas présent dans le jeu de données,
aucun service n'est créé.
</td>
</tr>
</table>


Utilisation dans le module {#arcanedoc_caseoptions_use}
===========================

Les options du jeu de données s'utilisent de manière naturelle dans le code.
Pour continuer l'exemple présenté en début de document, 
on souhaite avoir deux options
<tt>simple-real</tt> et <tt>boundary-condition</tt>. Par exemple :

~~~~~~~~~~~~~~~~{.cpp}
namespace TypesTest
{
  enum eBoundaryCondition
  {
    VelocityX,
    VelocityY,
    VelocityZ
  };
};
~~~~~~~~~~~~~~~~

Le bloc descripteur de module sera le suivant :

~~~~~~~~~~~~~~~~{.xml}
<?xml version="1.0" ?>
<module name="Test" version="1.0">
 <name lang='fr'>test</name>
 <description>Module de test</description>

 <variables/>
 <entry-points/>

 <!-- Liste des options -->
 <options>

   <simple name="simple-real" type="real">
     <description>Réel simple</description>
   </simple>

   <enumeration name="boundary-condition" type="TypesTest::eBoundaryCondition" default="X">
     <description>Type de condition aux limites</description>
     <enumvalue name="X" genvalue="TypesTest::VelocityX" />
     <enumvalue name="Y" genvalue="TypesTest::VelocityY"  />
     <enumvalue name="Z" genvalue="TypesTest::VelocityZ"  />
   </enumeration>

 </options>
</module>
~~~~~~~~~~~~~~~~

A partir de ce fichier, %Arcane va générer un fichier *Test_axl.h* 
qui contient, entre autre, une classe équivalente à celle-ci :

~~~~~~~~~~~~~~~~{.cpp}
class CaseOptionsTest
{
  public:
   ...
   double simpleReal() { ... }
   eBoundaryCondition boundaryCondition() { ... }
   ...
};
~~~~~~~~~~~~~~~~

Le module \c TestModule qui, par définition, hérite de la classe
\c ArcaneTestObject (voir \ref arcanedoc_module), peut
lire ses options en récupérant une instance de la classe \c CaseOptionsTest
grâce à la méthode \c options(). Après la lecture du jeu de données 
par %Arcane, le module pourra accéder à ses options par leur nom.

Par exemple, dans le module de test \c TestModule, on peut accéder 
aux options de la manière suivante :
  
~~~~~~~~~~~~~~~~{.cpp}
void TestModule::
myInit()
{
  if (options()->simpleReal() > 1.0)
    ...
  if (options()->boundaryCondition()==TypesTest::VelocityX)
    ...
}
~~~~~~~~~~~~~~~~

La partie du jeu de données concernant ce module peut être par exemple :

~~~~~~~~~~~~~~~~{.xml}
<test>
  <simple-real>3.4</simple-real>
  <boundary-condition>Y</boundary-condition>
</test>
~~~~~~~~~~~~~~~~

Gestion des valeurs par défaut {#arcanedoc_caseoptions_defaultvalues}
===============================

Il est possible de spécifier une valeur par défaut dans le fichier
**axl** pour les options simples, énumérées, étendues et les
services. Depuis la version 2.10.0 (septembre 2018), il est aussi possible
de définir ces valeurs par catégorie et de spécifier la catégorie
voulue lors de l'exécution. Le choix de la catégorie doit se faire
avant la lecture des options du jeu de donnée, par exemple dans la
classe gérant la session du code:
~~~~~~~~~~~~~~~~{.cpp}
#include "arcane/ICaseDocument.h"
using namespace Arcane;
void f()
{
  ISubDomain* sd = ...;
  ICaseDocument* doc = sd->caseDocument();
  doc->setDefaultCategory("MyCategory");
}
~~~~~~~~~~~~~~~~

Enfin, depuis la version 2.9.1 (juin 2018) de %Arcane, il est aussi possible
de changer dynamiquement lors de l'exécution ces valeurs par
défaut. Cela peut être utile par exemple si on veut des valeurs par
défaut par type de boucle en temps, par dimension du maillage, ...

Pour changer les valeurs par défaut, il existe une méthode
**setDefaultValue()** suivant le type de l'option:

| Classe %Arcane                                  |  Description
|-------------------------------------------------|---------------------------------------
| Arcane::CaseOptionSimpleT::setDefaultValue()    | options simples
| Arcane::CaseOptionEnumT::setDefaultValue()      | options énumérées
| Arcane::CaseOptionExtendedT::setDefaultValue()  | options étendues
| Arcane::CaseOptionService::setDefaultValue()    | services

\note Il n'est pas possible de changer les valeurs par défaut des options
possédant des occurences multiples.

Pour bien comprendre comment utiliser cette méthode, il est nécessaire
de connaitre les mécanismes de lecture des options du jeu de
données. La lecture du jeu de données se fait en plusieurs phases:
1. Phase 1. Lors de cette phase, on lit toutes les options sauf les
   options étendues car elles peuvent reposer sur le maillage et dans
   cette phase le maillage n'est pas encore lu. C'est lors de cette
   phase que sont aussi créés les différentes instances des services
   qui apparaissent dans le jeu de données.
2. Appel des points d'entrée **Build** du code.
3. Phase 2. Lors de cette phase, le maillage a été lu et on lit les
   options étendues du jeu de données. Après cette phase, toutes les
   options ont été lues.
4. Affichage dans le listing des valeurs des options du jeu de
   données.
5. Appel des points d'entrée **Init** du code.

Pour exécuter du code lors des parties (*2*) et (*5*), il faut utiliser des
points d'entrées déclarées dans la boucle en temps. Pour exécuter du code lors des parties 1 ou 3, il
est possible de s'enregistrer auprès du Arcane::ICaseMng::observable()
pour être notifié du début des phases 1 et 2. Le code sera exécuté
avant que %Arcane n'effectue la phase correspondante. Par exemple:

~~~~~~~~~~~~~~~~{.cpp}
#include "arcane/ObserverPool.h"
using namespace Arcane;
class MyService
{
 public:
  MyService(const ServiceBuildInfo& sbi)
  {
    ICaseMng* cm = sbi.subDomain()->caseMng();
    m_observers.addObserver(this,&MyService::onBeforePhase1,
                            cm->observable(eCaseMngEventType::BeginReadOptionsPhase1));
    m_observers.addObserver(this,&MyService::onBeforePhase2,
                            cm->observable(eCaseMngEventType::BeginReadOptionsPhase2));
  }
  void onBeforePhase1() { ... }
  void onBeforePhase2() { ... }
 private:
  ObserverPool m_observers;
};
~~~~~~~~~~~~~~~~

Les points suivants sont à noter:

- si on souhaite changer la valeur par défaut d'un service, il faut le
  faire lors de la partie (*1*) car ensuite les services ont déjà été
  créés.
- si une valeur par défaut est présente dans le fichier **axl**, ce
  sera cette valeurs qui sera utilisée tant qu'il n'y a pas eu d'appel
  à setDefaultValue(). Si on change une valeur d'une option simple
  lors de la partie (*3*) par exemple, elle ne sera pas encore prise
  en compte dans lors de l'appel des points d'entrée **Build** (qui
  sont dans la partie (*2*).
- il est possible de mettre une valeur par défaut même s'il n'y en a
  pas dans le fichier **axl**. Dans ce cas, il faut la positionner
  dans la partie (*1*) sinon %Arcane considérera le jeu de donnée
  comme invalide après lecture de la phase 1.
- si on souhaite changer une valeur par défaut en fonction des
  informations du maillage, il faut le faire lors de la partie (*3*).
