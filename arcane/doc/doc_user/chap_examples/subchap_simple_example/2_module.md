# Module SayHello {#arcanedoc_examples_simple_example_module}

[TOC]

Our SayHello module is therefore composed of the three files below:
- a header (.h),
- a source file (.cc),
- a file containing the dataset options (.axl).

\note By convention, module file names follow a precise scheme:
- the module name is free,
- the headers use the .h extension (and not .hh/.hpp/.hxx/etc.) and are composed
  as follows: {module_name} Module .h (example: SayHelloModule.h),
- the source files use the .cc extension (and not .c/.cxx/.c++etc.) and are
  composed as follows: {module_name} Module .cc (example: SayHelloModule.cc),
- the dataset configuration files (AXL files) use the .axl extension and are
  composed as follows: {module_name} .axl (example: SayHello.axl).

## SayHello.axl {#arcanedoc_examples_simple_example_module_sayhelloaxl}

First, let's look at the .axl file:

```xml
<?xml version="1.0" ?>
<module name="SayHello" version="1.0">

  <description>Descripteur du module SayHello</description>

  <variables>
    <variable
      field-name="loop_sum"
      name="LoopSum"
      data-type="integer"
      item-kind="none"
      dim="0"/>
  </variables>

  <entry-points>
    <entry-point method-name="startInit" name="StartInit" where="start-init"
                 property="none"/>
    <entry-point method-name="compute" name="Compute" where="compute-loop"
                 property="none"/>
    <entry-point method-name="endModule" name="EndModule" where="exit"
                 property="none"/>
  </entry-points>

  <options>
    <simple name="nSteps" type="integer" default="10">
      <description>Nombre de boucles à effectuer.</description>
    </simple>
  </options>

</module>
```
This is an XML format file that allows describing the functionality of our
`SayHello` module.

We can see that this file contains the module name (line 2), a brief
description (line 4), variables (lines 6-13), entry points (lines 15-19), and
options (lines 21-25).

Variables:
- The variable named "Arcane" `LoopSum` and named "code" `loop_sum`, of type
  `Integer`, assigned to the `None` item and with dimension `0`.

Entry points:
- The entry point named `StartInit`, represented by the `startInit` method and
  executed during initialization (`start-init`),
- The entry point named `Compute`, represented by the `compute` method and
  executed in the time loop (`compute-loop`).
- The entry point named `EndModule`, represented by the `endModule` method and
  executed when the time loop is finished (`exit`).

Options:
- The simple option named `nSteps`, of type `Integer` and with a default value
  equal to `10`.

The compiler, based on this file, will automatically create a `SayHello_axl.h`
file that must be imported into our header and which will allow us to use:
- The attribute `m_loop_sum` (`m_` + the `field-name`),
- The `startInit` method,
- The `compute` method,
- The `nStep` option.

## SayHelloModule.h {#arcanedoc_examples_simple_example_module_sayhellomoduleh}

Here is what our header looks like:

```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef SAYHELLOMODULE_H
#define SAYHELLOMODULE_H
 
#include <arcane/ITimeLoopMng.h>
#include "SayHello_axl.h"
 
using namespace Arcane;
 
class SayHelloModule
: public ArcaneSayHelloObject
{
 public:
  explicit SayHelloModule(const ModuleBuildInfo& mbi) 
  : ArcaneSayHelloObject(mbi) { }

 public:
  void startInit() override;
  void compute() override;
  void endModule() override;
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};
 
#endif
```

Let's go into detail.
```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
```
In chapter \ref arcanedoc_general_codingrules, we are asked to include this line
informing us of the file configuration.

____

```cpp
#ifndef SAYHELLOMODULE_H
#define SAYHELLOMODULE_H
//...
#endif
```
This prevents the class from being defined multiple times (if it is included in
several .cc files, for example).

____

```cpp
#include <arcane/ITimeLoopMng.h>
#include "SayHello_axl.h"
```
This first #include allows including the functions for managing the time loop
(for example `stopComputeLoop()`).
The second #include is the file generated from the .axl.

____

```cpp
using namespace Arcane;
```
This allows using %Arcane functions without prefixing them with `Arcane::`.

____

```cpp
class SayHelloModule
: public ArcaneSayHelloObject
```
We define the `SayHelloModule` class, which will be used by %Arcane and inherits
from `ArcaneSayHelloObject`. `ArcaneSayHelloObject` is defined in
`SayHello_axl.h` and contains the methods that can be overridden and the
variables/options that can be used.

____

```cpp
 public:
  explicit SayHelloModule(const ModuleBuildInfo& mbi) 
  : ArcaneSayHelloObject(mbi) { }
```
Constructor of our class, which calls the constructor of `ArcaneSayHelloObject`.
`mbi` is an object that will contain the module launch information (to get the
option values, for example).

____

```cpp
 public:
  void startInit() override;
  void compute() override;
  void endModule() override;
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
```
Finally, the four methods from the .axl that we override. We can also give a
version to our module by "overriding" `versionInfo()`.

____

## SayHelloModule.cc {#arcanedoc_examples_simple_example_module_sayhellomodulecc}

This file simply contains the implementations of the methods defined in the
header.

```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

#include "SayHelloModule.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SayHelloModule::
startInit()
{
  info() << "Module SayHello INIT";
  m_loop_sum = 0;
}

void SayHelloModule::
compute()
{
  info() << "Module SayHello COMPUTE";

  m_loop_sum = m_loop_sum() + m_global_iteration();

  if (m_global_iteration() > options()->getNStep())
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

void SayHelloModule::
endModule()
{
  info() << "Module SayHello END";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_SAYHELLO(SayHelloModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
```

Several things are noteworthy here:
```cpp
info() << "Module SayHello INIT";
```
`info()` allows writing information to the standard output.

____

```cpp
m_loop_sum = 0;
```
We assign the value 0 to the variable `m_loop_sum`.

____

```cpp
m_loop_sum = m_loop_sum() + m_global_iteration();
```
We calculate the sum of iterations.
\note
This is a variable of type `Arcane::VariableScalarInt32`. Therefore, you must
use the `()` operator to retrieve its value.
\warning
We modify the `m_loop_sum` variable at each iteration for the example. In
reality, using assignment for a variable of type `Arcane::VariableRefScalarT`
can be costly.

____

```cpp
if (m_global_iteration() > options()->getNStep())
  subDomain()->timeLoopMng()->stopComputeLoop(true);
```
`m_global_iteration()` is a variable of the `CommonVariables` class, from which
the `BasicModule` class inherits (and `ArcaneSayHelloObject` inherits from it,
from the `SayHello_axl.h` file). This variable contains the currently running
iteration.
\note
This is also a variable of type `Arcane::VariableScalarInt32`. Therefore, you
must use the `()` operator to retrieve its value.

`options()->getNStep()` allows retrieving the value of the `nStep` option. To
define this value, you must use the dataset (the .arc file) (see the next
section \ref arcanedoc_examples_simple_example_arc).

Finally, `subDomain()->timeLoopMng()->stopComputeLoop(true)` allows stopping the
time loop.

____

```cpp
ARCANE_REGISTER_MODULE_SAYHELLO(SayHelloModule);
```
This is a macro that allows registering the module in %Arcane. This macro is
defined in the `SayHello_axl.h` file.

____

\remarks
As you can see, in the three files above, there is no mention of the
HelloWorld application.
Modules can be easily placed and replaced in other applications.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_struct
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_arc
</span>
</div>
