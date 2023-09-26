# Le support des tests unitaires dans %Arcane {#arcanedoc_debug_perf_unit_tests}

[TOC]

## Introduction {#arcanedoc_debug_perf_unit_tests_intro}

Les tests unitaires dans %Arcane sont exécutés sans que la simulation
ne soit lancée. Ils permettent de tester des services, pas des
modules. Leur mise en oeuvre est simple et rapide : les services
déclarent des méthodes de test dans le descripteur (fichier
'.axl'). Ces méthodes sont exécutées par un module de test et une
boucle en temps dédiés qui sont fournis par %Arcane. Il suffit, pour
exécuter les tests, d'ajouter le service à la liste des services du
module de test.
  
Cette page décrit les différentes étapes à suivre pour construire et
exécuter des tests unitaires. Notons, qu'à ce jour, les tests unitaires
ne fonctionnent qu'en séquentiel. La génération du rapport de test
n'est pas encore implémentée en parallèle.
  
## Déclaration des tests {#arcanedoc_debug_perf_unit_tests_decl}

Les tests unitaires peuvent être ajoutés à n'importe quel service. Il
suffit pour cela de déclarer les méthodes de test dans le descripteur
du service (fichier 'axl') sous la forme suivante :
  
```xml
<tests class-set-up="setUpForClass" test-set-up="setUp" class-tear-down="tearDownForClass" test-tear-down="tearDown">
  <test name="Test 1" method-name="myTestMethod1"/>
  <test name="Test 2" method-name="myTestMethod2"/>
  <test name="Test 3" method-name="myTestMethod3"/>
</tests> 
```

Le descripteur ci-dessus déclare :

- 3 méthodes de test 'myTestMethod1', 'myTestMethod2' et
  'myTestMethod3'. Ces 3 tests comportent un nom (attribut 'name') qui
  est utilisé pour les affichages lors de l'exécution du test.
- 2 méthodes optionnelles 'setUp' et 'tearDown' qui sont appelées
  respectivement avant et après chaque appel à une méthode de test. Si n
  méthodes de test sont déclarées, 'setUp' et 'tearDown' sont donc
  appelées n fois. Notons qu'il est tout à fait possible de ne déclarer
  que le setUp ou que le tearDown.
- 2 méthodes optionnelles 'setUpForClass' et 'tearDownForClass' qui
  sont appelées respectivement avant et après l'exécution de
  l'ensemble des méthodes déclarées dans le descripteur. Quel que soit
  le nombre de méthodes déclarées, 'setUpForClass' et
  'tearDownForClass' ne sont donc appelées qu'une seule fois.

Une fois ce travail effectué, il vous reste à écrire les méthodes
déclarées dans le descripteur dans votre code du service. Pour
l'exemple précédent, on obtient pour le fichier '.h' :
  
```cpp
 ...
public:
 void setUpForClass();
 void tearDownForClass();
 void setUp();
 void tearDown();
 void myTestMethod1();
 void myTestMethod2();
 void myTestMethod3();
 ...
```
  
Néanmoins, si vous oubliez de définir une des méthodes dans votre '.h', une erreur de compilation se produira. 
Dans l'exemple précédent, si vous oubliez de définir 'myTestMethod1', vous obtiendrez le message suivant :

```
error: no 'void MonServiceDeTest::myTestMethod1()' member function declared in class 'MonServiceDeTest'
```
  
## Les assertions {#arcanedoc_debug_perf_unit_tests_assertions}

Il faut maintenant coder les méthodes de test. Comme pour la plupart
des bibliothèques de tests unitaires (CppUnit, GoogleTest...), %Arcane
met à disposition un ensemble d'assertions pour tester les résultats
de tests. Ces assertions sont disponibles sous forme de macros C++.

\`A ce jour, les macros disponibles sont :

- FAIL : qui permet de faire échouer un test. Cette macro est utile,
  par exemple, pour vérifier qu'une exception est appelée. Il suffit
  de faire un appel à FAIL après l'instruction qui doit déclencher
  l'exception. Si l'exception n'est pas déclenchée, l'instruction
  suivante est exécutée et FAIL est appelée.
- ASSERT_TRUE(condition) : qui permet de vérifier qu'une valeur booleéenne est vraie. Par exemple, ASSERT_TRUE(i<5).
- ASSERT_FALSE(condition) : assertion inverse à la précédente.
- ASSERT_EQUAL(expected, actual) : permet de vérifier l'égalité entre la valeur attendue pour le test et le résultat effectivement obtenu.
  Cette macro s'appuie sur une méthode générique (template) qui utilise l'opérateur '=='. Cette macro est donc valable pour tout type 
  définissant cet opérateur. Notons que les types de base de Arcane répondent à cette exigence (Integer, Real, Real2...).
- ASSERT_NEARLY_EQUAL(expected, actual) : permet de vérifier la 'presque' égalité entre la valeur attendue pour le test et le résultat effectivement obtenu.
  Cette assertion est utile pour tester l'égalité entre des réels malgré les imprécisions machine.
  Cette macro s'appuie sur une méthode générique (template) qui utilise la méthode 'math::isNearlyEqual'. Cette macro est donc valable pour tout type 
  définissant cette méthode. C'est le cas des réels Arcane.
- ASSERT_NEARLY_EQUAL_EPSILON(expected, actual, epsilon) : fonctionne comme l'assertion précédente avec un epsilon de comparaison fourni en paramètre par l'appelant.
  
Voici quelques exemples d'utilisation de ces macros :

```cpp
ASSERT_TRUE(i <= 5);
ASSERT_FALSE(i > 5);
ASSERT_EQUAL(5, x);
ASSERT_NEARLY_EQUAL(5.5, y);
```

### Tests unitaires en parallèle {#arcanedoc_debug_perf_unit_tests_parallel}

Depuis la version 2.20 de %Arcane, il est possible d'utiliser les
tests unitaires en parallèle. Pour cela, une version des macros est
disponible en spécifiant en paramètre une instance de
Arcane::IParallelMng. Ces macros sont identiques dans leurs sémantiques
à la version séquentielle et sont préfixées par `PARALLEL_`, comme par
exemple PARALLEL_ASSERT_TRUE ou PARALLEL_ASSERT_NEARLY_EQUAL. Ces
appels sont collectifs et le test est considéré comme ayant échoué si
un des rangs a échoué.

Voici un exemple d'utilisation :

```cpp
using namespace Arcane;
IParallelMng* pm = ...;
Real deltat = ...;
PARALLEL_ASSERT_NEARLY_EQUAL(deltat,1.0,pm);
```

### Cas particulier des exceptions {#arcanedoc_debug_perf_unit_tests_exception} 

Parfois, on souhaite développer un test unitaire pour vérifier qu'une méthode déclenche une exception.
Supposons qu'on veuille faire un test qui déclenche MyException dans la méthode myMethod().
La technique consiste à appeler la macro FAIL juste après myMethod(). Ainsi, si l'exception est déclenchée,
la macro n'est pas appelée et le test passe. Il suffit alors de traiter
l'exception dans un bloc catch.  

```cpp
try {
  myMethod();   // doit déclencher mon exception...
  FAIL;         // ... si je suis ici c'est que l'exception n'est pas déclenchée
} catch (const MyException& e) {
  // ok, l'exception a été appelée
}
```

Parfois la méthode ne lève pas une exception mais appelle la méthode
TraceAccessor::fatal() du gestionnaire de traces de Arcane (cf. \ref
arcanedoc_execution_traces). Cette méthode lève une exception de type
Arcane::FatalErrorException qu'il suffit de traiter comme dans
l'exemple ci-dessus.

## Le fichier de données {#arcanedoc_debug_perf_unit_tests_data}

Les tests unitaires s'exécutent grâce à un service et à une boucle en
temps spécifiques fournis par %Arcane. La seule chose à faire est de
sélectionner la boucle en temps en question dans le fichier de données
du code et d'ajouter votre service de test à la liste des services du module.
  
L'exemple ci-dessous montre un fichier de données type avec 'MonServiceDeTest' dans la liste des services de test.

```xml
 <arcane>
   <titre>Mon cas test</titre>
   <description>Description de mon cas test</description>
   <boucle-en-temps>UnitTest</boucle-en-temps>
 </arcane>
 ...
 <module-test-unitaire>
   ...
   <xml-test name="MonServiceDeTest">
     <!-- ici les données de mon service de test (si bien sur il en a !)... -->
   </xml-test>
   ...
 </module-test-unitaire>
```

## L'exécution {#arcanedoc_debug_perf_unit_tests_run}
  
Il suffit alors d'exécuter votre programme comme d'habitude en lui
donnant comme jeu de données, le fichier défini à l'étape
précédente. Le module de test va alors effectuer une itération en
déclenchant l'ensemble des méthodes de test des services fournis dans
le fichier de données. Vous pourrez alors voir dans le listing la
trace de l'exécution des tests comme ci-dessous.
   
On remarque que le listing signale également le chemin du rapport de test.
  
```
...
*I-Master     *** ITERATION        1  TEMPS 1.000000000000000e+00  BOUCLE        1  DELTAT 1.000000000000000e+00 ***
*I-Master     Date: 2013-05-28T14:52:28 Conso=(R=0,I=0,C=0) Mem=(62,m=62:0,M=62:0,avg=62)
*I-UnitTest   [OK   ] myTestMethod1
*I-UnitTest   [ECHEC] myTestMethod2 (line 136 in virtual void MonServiceDeTest::myTestMethod2())
*I-UnitTest           Obtenu : 6.5. Attendu : 5.5.
*I-UnitTest   [OK   ] myTestMethod3
...
*I-UnitTest   Sortie du rapport de test unitaire dans '/tmp/moncas/output/listing/unittests.xml'
...
```

Le listing est un fichier XML. Il a le format suivant :
  
```xml
<unit-tests-results>
  <service name="MonServiceDeTest">
    <unit-test method-name="Test 1" name="myTestMethod1" result="success" />
    <unit-test method-name="Test 2" name="myTestMethod2" result="failure">
       <exception file="/tmp/monprojet/src/MonServiceDeTest.cc" 
               line="136" message="Obtenu : 6.5. Attendu : 5.5."
		 where="virtual void MonServiceDeTest::myTestMethod2()" />
    </unit-test>
    <unit-test method-name="Test 3" name="myTestMethod3" result="success" />
	</service>
</unit-tests-results>
```

## Utiliser sa propre boucle en temps {#arcanedoc_debug_perf_unit_tests_own_timeloop}

Parfois, avant d'exécuter les tests unitaires, on aimerait effectuer
des initialisations réalisées dans des points d'entrée des modules de
l'application. Or, comme les tests unitaires sont exécutés dans une
boucle en temps spécifique, ces points d'entrée ne sont pas appelés.

Pour contourner cela, il est possible de faire sa propre boucle en
temps. Cette boucle en temps doit obligatoirement comporter les 3 points d'entrée suivants :

- UnitTest.UnitTestInit dans la section Init,
- UnitTest.UnitTestDoTest dans la section ComputeLoop,
- UnitTest.UnitTestExit dans la section Exit.

Il faut penser à ajouter le module 'UnitTest' à la liste des modules de la boucle.

Il suffit alors d'insérer ses propres points d'entrée d'initialisation
avant 'UnitTest.UnitTestInit'. La section 'ComputeLoop' ne comporte
généralement que le point d'entrée 'UnitTest.UnitTestDoTest'.

```xml
<arcane-config code-name="MyCode">
  <time-loops>
    <time-loop name="MyTimeLoop">
      <title>My nice timeloop</title>
      <modules>
        <module name="UnitTest" need="required" />
        ...
      </modules>

      <entry-points where="init">
        ...
        <entry-point name="UnitTest.UnitTestInit" /> 
      </entry-points>

      <entry-points where="compute-loop">
        <entry-point name="UnitTest.UnitTestDoTest" /> 
      </entry-points>

      <entry-points where="exit">
        <entry-point name="UnitTest.UnitTestExit" /> 
      </entry-points>
    </time-loop>
  </time-loops>
</arcane-config>
```

Notons qu'il ne faut pas oublier de changer le nom de la boucle en
temps dans le fichier de données !


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_compare_bittobit
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_profiling
</span>
</div>
