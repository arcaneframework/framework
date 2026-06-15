# Using C# functions for the dataset {#arcanedoc_wrapping_csharp_casefunction}

[TOC]

Starting from version 3.11 of %Arcane, it is possible to define dataset
functions in C#. There are two types of functions:

- classic functions automatically used by %Arcane
- advanced functions that must be explicitly called by the developer.

Example \ref arcanedoc_sample_userfunction shows how to use user functions in
C#.

## Simple Functions

To define dataset functions in C#, the user must define a class containing a set
of public methods whose signature matches that of a dataset function. In this
case, %Arcane generates a dataset function whose name is that of the
corresponding method.

Valid signatures are those corresponding to methods that take an argument of
type `double` or `Int32` and return a type of `double` or `Int32`.

For example, with the following code, there will be 2 user functions

```{cs}
public class MyDotNetFunctions
{
  // Fonction utilisateur valide
  public double Func1(double x)
  {
    return x * 2.0;
  }
  // Fonction utilisateur valide
  public double Func1(int x)
  {
    return (double)x *2.0;
  }
  // Signature ne correspondant pas. La méthode sera ignorée
  public void SampleMethod(int x)
  {
  }
}
```

## Advanced Functions

Advanced functions must be called directly by C++ code. The
\arcane{IStandardFunction} interface allows retrieving a pointer to these
methods. There are 4 possible prototypes:

- f(Real,Real) -> Real
- f(Real,Real3) -> Real
- f(Real,Real) -> Real3
- f(Real,Real3) -> Real3

For a simple dataset option, it is possible to retrieve the
\arcane{IStandardFunction} instance via the
\arcane{CaseOptionSimple::standardFunction()} method.

For example, if the dataset option `node-velocity` must have a function with the
signature `f(Real,Real3) -> Real3`, we can retrieve the instance and use it as
follows:

~~~{cpp}
Arcane::IBinaryMathFunctor<Real, Real3, Real3>* functor = nullptr;
Arcane::IStandardFunction* scf = options()->nodeVelocity.standardFunction();
if (!scf)
  ARCANE_FATAL("No standard case function for option 'node-velocity'");
functor = scf->getFunctorRealReal3ToReal3();
if (!functor)
  ARCANE_FATAL("Standard function is not convertible to f(Real,Real3) -> Real3");

// Call the user function
Arcane::Real3 position(1.2,0.4,1.5);
functor->apply(1.2,position);
~~~

The `functor` instance is not modified during the calculation. It is therefore
possible to keep it between iterations.

## Compilation and Usage

The `arcane_dotnet_compile` command, available in the `bin` directory of the
%Arcane installation, allows compiling a C# file into an assembly (`.dll`) which
will be placed in the current directory. The name of the assembly is the name of
the compiled file without the `.cs` extension. The compiled file is
platform-independent, so it only needs to be compiled once.

```{sh}
> ls .
Functions.cs
> arcane_dotnet_compile Functions.cs
> ls .
Functions.cs  Functions.dll  Functions.pdb
```

You must then reference this assembly in the `.arc` file within the
`<functions>` element at the root of the dataset:

```{xml}
<functions>
  <external-assembly>
    <assembly-name>Functions.dll</assembly-name>
    <class-name>MyDotNetFunctions</class-name>
  </external-assembly>
</functions>
```

When running the calculation, you must specify the argument `-A,UsingDotNet=1`
so that the `.Net` environment is used and the `Functions.dll` assembly is
loaded. If the assembly name is relative (for example, on Linux it does not
start with `/`), the assembly must be located in the current execution
directory.

Once the assembly is loaded, %Arcane will create an instance of the class
specified in the `<class-name>` element. It is therefore possible to have a
constructor in this class, and it will be used. The class may be located in a
`namespace`. In this case, you must specify the full name in `<class-name>`,
such as `MyNamespace.MyClass`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_wrapping_csharp_swig
</span>
<span class="next_section_button">
\ref arcanedoc_wrapping_python
</span>
</div>
