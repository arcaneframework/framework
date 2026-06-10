# Usage in the module {#arcanedoc_core_types_axl_caseoptions_usage}

[TOC]

The dataset options are used naturally in the code.
To continue the example presented at the beginning of the document, we want to
have two options:
<tt>simple-real</tt> and <tt>boundary-condition</tt>. For example:

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

The module descriptor block will be the following:

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

    <enumeration name="boundary-condition" type="TypesTest::eBoundaryCondition"
                 default="X">
      <description>Type de condition aux limites</description>
      <enumvalue name="X" genvalue="TypesTest::VelocityX"/>
      <enumvalue name="Y" genvalue="TypesTest::VelocityY"/>
      <enumvalue name="Z" genvalue="TypesTest::VelocityZ"/>
    </enumeration>

  </options>
</module>
```

From this file, %Arcane will generate a file *Test_axl.h* which contains, among
other things, a class equivalent to this one:

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

The \c TestModule module, which by definition inherits from the
\c ArcaneTestObject class (see \ref arcanedoc_core_types_module), can read its
options by retrieving an instance of the \c CaseOptionsTest class using the
\c options() method. After %Arcane reads the dataset, the module can access its
options by their name.

For example, in the \c TestModule test module, options can be accessed in the
following way:

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

The dataset part concerning this module can be, for example:

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
