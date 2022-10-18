# Utilisation dans le module {#arcanedoc_core_types_axl_caseoptions_usage}

[TOC]

Les options du jeu de données s'utilisent de manière naturelle dans le code.
Pour continuer l'exemple présenté en début de document, 
on souhaite avoir deux options
<tt>simple-real</tt> et <tt>boundary-condition</tt>. Par exemple :

```cpp
namespace TypesTest
{
  enum eBoundaryCondition
  {
    VelocityX,
    VelocityY,
    VelocityZ
  };
};
```

Le bloc descripteur de module sera le suivant :

```xml
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
```

A partir de ce fichier, %Arcane va générer un fichier *Test_axl.h* 
qui contient, entre autre, une classe équivalente à celle-ci :

```cpp
class CaseOptionsTest
{
  public:
   ...
   double simpleReal() { ... }
   eBoundaryCondition boundaryCondition() { ... }
   ...
};
```

Le module \c TestModule qui, par définition, hérite de la classe
\c ArcaneTestObject (voir \ref arcanedoc_core_types_module), peut
lire ses options en récupérant une instance de la classe \c CaseOptionsTest
grâce à la méthode \c options(). Après la lecture du jeu de données 
par %Arcane, le module pourra accéder à ses options par leur nom.

Par exemple, dans le module de test \c TestModule, on peut accéder 
aux options de la manière suivante :
  
```cpp
void TestModule::
myInit()
{
  if (options()->simpleReal() > 1.0)
    ...
  if (options()->boundaryCondition()==TypesTest::VelocityX)
    ...
}
```

La partie du jeu de données concernant ce module peut être par exemple :

```xml
<test>
  <simple-real>3.4</simple-real>
  <boundary-condition>Y</boundary-condition>
</test>
```


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions_options
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions_default_values
</span>
</div>