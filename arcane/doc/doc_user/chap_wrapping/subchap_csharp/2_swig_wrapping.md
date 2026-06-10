# C# Extensions with Swig {#arcanedoc_wrapping_csharp_swig}

[TOC]

Swig is an open-source tool that allows interfacing C/C++ code with another
language. In the case of Arcane, we wrap C++ classes in C# so that they can be
used in any language using '.Net'.

At least version 4.0 of Swig is required.

Wrapping is done by describing in a file with the .i extension which files will
be wrapped and how to do it. Generally, a C++ class will have a C# class of the
same name. By convention, C# methods do not start with a capital letter. To
respect this, the methods of classes wrapped by Arcane are converted. For
example, the C++ method Arcane::ISubDomain::caseMng() will become the C# method
`CaseMng()`.

The class `Arcane::String` is converted into the `.Net` class `string`.

Starting from the '.i' file, `swig` will generate a `C++` file and a set of `C#`
files. The former must be compiled as normal `C++` code in the form of a dynamic
library, and the latter as a standard `C#` project. Communication between `C++`
and `C#` is done via a `.Net` mechanism called PInvoke (for Platform Invoke).
During execution, the library compiled from C++ must be accessible. For this, it
must be in the same directory as the `C#` assembly or accessible via environment
variables (LD_LIBRARY_PATH on Unix or PATH on Windows).

The 'eos/csharp' example shows how to perform the wrapping and run the C# code.

%Arcane uses the `swig` tool to make different classes accessible in `C#`.

This document describes how the developer can add their own classes so that they
are accessible in `C#`.

The `swig` tool uses a file with the `.i` extension to describe the classes to
be wrapped.

For example, let's assume we have a C++ interface representing the interface of
an equation of state calculation service and that we want to be able to
implement this service in `C#`. The interface is defined in the file
`IEquationOfState.h`:

```cpp
using namespace Arcane;

namespace EOS
{
//! Interface du service du modèle de calcul de l'équation d'état.
class IEquationOfState
{
 public:
  /** Destructeur de la classe */
  virtual ~IEquationOfState() = default;
  
 public:
  /*!
   *  Initialise l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et l'énergie interne. 
   */
  virtual void initEOS(const CellGroup& group,
                       const VariableCellReal& pressure,
                       const VariableCellReal& adiabatic_cst,
                       const VariableCellReal& density,
                       VariableCellReal& internal_energy,
                       VariableCellReal& sound_speed
                       ) =0;
  /*!
   *  Applique l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et la pression. 
   */
  virtual void applyEOS(const CellGroup & group,
                        const VariableCellReal& adiabatic_cst,
                        const VariableCellReal& density,
                        const VariableCellReal& internal_energy,
                        VariableCellReal& pressure,
                        VariableCellReal& sound_speed
                        ) = 0;
};

}
```

The interface defines two methods, `initEOS` and `applyEOS`, which we want to
make accessible in `C#`.

To do this, you must define a file `EOSCSharp.i` containing the following code:

```i
// 1ère partie
%module(directors="1") EOSCharp

%import core/ArcaneSwigCore.i

%typemap(csimports) SWIGTYPE
%{
using Arcane;
%}

%{
#include "ArcaneSwigUtils.h"
#include "arcane/ServiceFactory.h"
#include "arcane/ServiceBuilder.h"
#include "IEquationOfState.h"
using namespace Arcane;
%}

/*---------------------------------------------------------------------------*/
// 2ème partie
ARCANE_DECLARE_INTERFACE(EOS,IEquationOfState)

%include IEquationOfState.h

/*---------------------------------------------------------------------------*/
// 3ème partie
ARCANE_SWIG_DEFINE_SERVICE(EOS,IEquationOfState,
                           public abstract void InitEOS(CellGroup group,
                                                        VariableCellReal pressure,
                                                        VariableCellReal adiabatic_cst,
                                                        VariableCellReal density,
                                                        VariableCellReal internal_energy,
                                                        VariableCellReal sound_speed);
                           public abstract void ApplyEOS(CellGroup group,
                                                         VariableCellReal adiabatic_cst,
                                                         VariableCellReal density,
                                                         VariableCellReal internal_energy,
                                                         VariableCellReal pressure,
                                                         VariableCellReal sound_speed);
                           );
```

This file has three parts:

1. The first part, common to all wrappers, which describes the module name and
   the code that will be integrated into the generated code. All `.h` files of
   the classes to be wrapped by `swig` must be placed in this part.
2. The second part, which explicitly indicates the classes to be wrapped. If a
   `C++` class must be considered a `C#` interface, you must use the macro
   `ARCANE_DECLARE_INTERFACE` to specify it. The first parameter is the name of
   the `namespace` and the second is the name of the class. Note that if you
   wish to define multiple interfaces, you must declare them all before any
   potential `%include`.
3. The third part, which defines the interfaces that we want to extend as a
   service. For this, we use the macro `ARCANE_SWIG_DEFINE_SERVICE`. This macro
   contains 3 arguments. The first two are equivalent to those of the
   `ARCANE_DECLARE_INTERFACE` macro. The last contains the `C#` signature of the
   wrapped interface's methods. While not mandatory in theory, it ensures that
   the user correctly implements the interface methods. Indeed, via the `swig`
   mechanism, this is not necessarily guaranteed, and if the user does not
   override the methods, it causes an error during execution. Swig will then
   generate a `DirectorPureVirtualException`.

## Compiling the code generated by swig {#arcanedoc_wrapping_csharp_swig_build}

`swig` will generate two types of files:

- the 'C++' files containing the wrapping, which will be called directly from
  `C#`.
- the `C#` files which contain the classes generated by `swig` and which will be
  used by the developer.

For the wrapping to work, you must compile both the C++ files as a dynamic
library and the `C#` files as an assembly.

\note For the .Net code to easily call native code (C/C++), the latter must be
accessible in the form of a dynamic library. It is mandatory to compile the C++
code generated by swig in the form of a dynamic library. To be precise, it is
possible to do this differently, but it requires writing code dependent on the
runtime (`mono` or `coreclr`) used.

If the developer uses `cmake`, there is a package that manages both the
generation via swig and creates a target for the corresponding C++ code. If our
`.i` file is named `Wrapper.i` and the target is `arcane_wrapper`, we can use
the following code:

```cmake
set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(UseSWIG_MODULE_VERSION 2)
include(UseSWIG)

swig_add_library(arcane_wrapper
  TYPE SHARED
  LANGUAGE CSHARP
  SOURCES Wrapper.i
)
```

\todo add info on the CMake package ArcaneSwigUtils.cmake

For the `C#` code, you must use a `C#` project file which will be compiled via
the `dotnet build` or `dotnet publish` command.





____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_wrapping_csharp_dotnet
</span>
<span class="next_section_button">
\ref arcanedoc_wrapping_csharp_casefunction
</span>
</div>
