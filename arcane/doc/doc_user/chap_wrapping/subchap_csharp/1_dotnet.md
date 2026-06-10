# Using C# with %Arcane {#arcanedoc_wrapping_csharp_dotnet}

[TOC]

Most of the %Arcane API is accessible via the .Net technology. It is possible to
write modules and services in C#.

The following pages are available:
- \ref arcanedoc_wrapping_csharp_swig : shows how to create an extension of C++
  classes so that they are accessible in `C#`.

## Using .Net {#arcanedoc_wrapping_csharp_dotnet_usage}

In 2019, there were two environments for .Net:

- The historical framework, called .Net Framework, which was originally
  developed for Windows but for which there is an open source implementation
  called `mono`.
- the new open source implementation called `coreclr` which uses a new framework
  called '.Net Core'. This implementation is available for Windows, Linux, and
  MacOS.

`.Net Core` and `.Net Framework` share a large part of the common APIs, and
there are generally no difficulties in using one or the other. The historical
framework `.Net Framework` will no longer evolve (but will continue to be
maintained), and all new features will be in the '.Net Core' framework. To
simplify the nomenclature, in 2020 there will only be one name, which will be
`.Net`.

%Arcane supports both the `mono` and `coreclr` implementations. The minimum
versions are `5.16` for `mono` and `3.0` for coreclr. For both implementations,
it is possible to run C# code either with a C# `main` or in embedded mode with a
C++ `main`.

The mode with the C# main allows the code to be launched like any C# code:

- with `mono`: `mono MyExe.dll`
- with `dotnet`: `dotnet MyExe.dll`

This mode is especially useful for debugging, for example with 'Visual Studio
Code' (TODO: provide example).

The embedded mode launches the code as a C++ executable, and it is the call to
Arcane::ArcaneLauncher::run() that will potentially load the '.Net' runtime and
load the necessary assemblies.

## .Net Basics {#arcanedoc_wrapping_csharp_dotnet_basics}

`.Net` is a technology quite similar to `java` in principle. The source code can
be written in several languages (C#, F#, Visual Basic). The code is compiled
into a platform-independent pseudo-assembly (bytecode). The product of this
compilation is called an *assembly* (equivalent to dynamic libraries in C++).
In '.Net', the extension is `.dll`, just like dynamic libraries (Dynamic Loaded
Library) under Windows.

Like Java, the bytecode is converted into code specific to the target machine's
architecture during execution. `.Net` code requires the presence of a runtime to
manage this part, as well as other features like the Garbage Collector.

### C# Main Example {#arcanedoc_wrapping_csharp_dotnet_example}

By convention, C# files have the `.cs` extension. The C# code is very similar to
C++ code:

```cs
using Arcane;
public class MainClass
{
  public static int Main(string args)
  {
    var cmd_line_args = CommandLineArguments.Create(args);
    ApplicationInfo app_info = ArcaneLauncher.ApplicationInfo;
    app_info.SetCommandLineArguments(cmd_line_args);
    app_info.SetCodeName("MyCode");
    app_info.SetCodeVersion(new VersionInfo(1,0,0));
    return ArcaneLauncher.Run();
  }
}
```

### Compilation {#arcanedoc_wrapping_csharp_dotnet_build}

`.Net` uses a tool called `msbuild` for compilation, and you must define a
project in XML format containing the necessary information.

\note It is also possible to compile directly in the manner of C++, but this
method is less portable because you must directly specify the necessary
assemblies.

`msbuild` uses a project file to define the compilation elements. In principle,
this project file is like the `Makefile` for the `make` tool or the
`CMakeLists.txt` for the `CMake` tool. In C#, for `msbuild`, this file
conventionally has the `.csproj` extension. Generally, a C# project is created
in a specific directory. The `dotnet new` command allows you to create a
directory with a project:

```sh
dotnet new console -n MyTest
```

This will create a `MyTest` directory containing a `Program.cs` file and a
`MyTest.csproj` file.

The `MyTest.csproj` file will be as follows:

```xml
 <Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>netcoreapp3.1</TargetFramework>
  </PropertyGroup>

</Project>
```

and the `Program.cs` file as follows:

```cs
using System;

namespace test2
{
  class Program
  {
    static void Main(string[] args)
    {
      Console.WriteLine("Hello World!");
    }
  }
}
```

To compile this file, simply navigate to the project directory and run the
`dotnet build` command.

\note By default, all files with the `.cs` extension present in the project
directory and subdirectories are compiled. This is why it is preferable to place
projects in subdirectories. To avoid this behavior, it is possible to set the
property *EnableDefaultCompileItems* to *false* and specifically add the files
to compile. For example:

```xml
<PropertyGroup>
  <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
</PropertyGroup>
<ItemGroup>
<Compile Include="MyFile1.cs"/>
<Compile Include="MyFile2.cs"/>
</ItemGroup>
```

The `dotnet build` command by default creates the assembly in the directory
`bin/${Config}/${framework}` with *Config* having the value *Debug* or *Release*
and *framework* having the value of the `msbuild` *TargetFramework* property. In
our example, the directory will therefore be `bin/Debug/netcoreapp3.1`.

To run the program, you must run the command:

```sh
dotnet bin/Debug/netcoreapp3.1/MyTest.dll
```

\note It is also possible to launch the execution directly using `dotnet run`.
But before launching the execution, this command will check if it is necessary
to recompile the program, which can take time. If you are sure you have not
modified anything, specifying the *dll* directly as an argument to `dotnet` is
preferable.

### Using a DLL in C# {#arcanedoc_wrapping_csharp_dotnet_dll}

If you do not want a C# executable but want to use C# code, the functioning is
almost identical, but instead of creating an executable, you must create a
library. To do this, simply create the C# project with the option
`dotnet new classlib -n MyLib`.

\note With 'Net Core 3', there is no fundamental difference between an
executable and a DLL. An executable is just a DLL with a well-defined entry
point (the `Main` function). In both cases, the extension will be `.dll`, and an
executable can be used wherever a DLL can be used.

### Adding References to Arcane NuGet Packages {#arcanedoc_wrapping_csharp_dotnet_refnuget}

Adding references to Arcane DLLs is done via the `<PackageReference>` element in
`msbuild`, for example:

```xml
<ItemGroup>
  <PackageReference Include="Arcane.Utils" Version="2.19.0" />
</ItemGroup>
```

The following `nuget` packages are provided by %Arcane:

- Arcane.Utils: contains the C# utility classes (arrays and views).
- Arcane.Core : contains the C# extensions of Arcane core classes (the C++
  classes whose `.h` files are in the 'arcane' directory).
- Arcane.Launcher : contains the C# classes for managing C# code launching. This
  package is useful if you want to have a C# main.
- Arcane.Services : contains the C# classes for implementing Arcane's basic
  services (Arcane::IDataWriter, Arcane::IDataReader, ...)
- Arcane.Hdf5: contains the C# classes corresponding to the C++ file '
  Hdf5Utils.h'. This package is available if %Arcane was compiled with `hdf5`
  support.
- Arcane.Cea.Materials : contains the C# classes managing
  materials/environments (CEA only).

### Extending C++ Classes with 'SWIG' {#arcanedoc_wrapping_csharp_dotnet_classesextends}

The page \ref arcanedoc_wrapping_csharp_swig describes how to make %Arcane
classes accessible in C# and extend other classes.

### Creating a .Net Library Using Arcane {#arcanedoc_wrapping_csharp_dotnet_libcreation}

First, you must have `dotnet` in your path. If %Arcane is installed in the
directory `${ARCANE_PREFIX}`, then you must run the following commands:

```sh
ARCANE_PREFIX=/path/to/arcane/install
mkdir /path/to/my/project
cd /path/to/my/project
dotnet new classlib
dotnet add package Arcane.Core --source ${ARCANE_PREFIX}/nupkgs
```

To compile the code and install it in a directory, you must execute the
following command:

```sh
dotnet publish -o out
```

This will result in the generated files being installed in the `out` directory.
If the project is named 'toto', you will have the following files:

```sh
> ls  out
Arcane.Core.dll  Arcane.Utils.dll  toto.deps.json  toto.dll  toto.pdb
```

### Creating a .Net Binary Using %Arcane {#arcanedoc_wrapping_csharp_dotnet_bincreation}

\todo To be written.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_wrapping_csharp
</span>
<span class="next_section_button">
\ref arcanedoc_wrapping_csharp_swig
</span>
</div>
