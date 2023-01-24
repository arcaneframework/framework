# Utilisation du C# avec %Arcane {#arcanedoc_wrapping_csharp_dotnet}

[TOC]

La plus grande partie de l'API %Arcane est accessible via la
technologie `.Net`. Il est possible d'écrire des modules et service en
C#.

Les pages suivantes sont disponibles :
- \ref arcanedoc_wrapping_csharp_swig : montre comment faire une extension
  de classes C++ pour qu'elles soient accessibles en `C#`.

## Utilisation de .Net {#arcanedoc_wrapping_csharp_dotnet_usage}

En 2019, il existe deux environnments pour `.Net`:

- Le framework historique, appelé `.Net Framework` qui est à l'origine
  développé pour Windows mais pour lequel il existe une implémentation
  open source appelée `mono`.
- la nouvelle implémentation open source appelée `coreclr` et qui
  utilise un nouveau framework appelé '.Net Core'. Cette
  implémentation est disponible pour Windows, Linux et MacOS.

`.Net Core` et `.Net Framework` partagent une grande parties des API
communes et il n'y en général pas de difficultés pour utiliser l'un ou
l'autre. Le framework historique `.Net Framework` n'évoluera plus
(mais continuera à être maintenu) et toutes les nouveautés se feront
dans le framework '.Net Core'. Pour simplifier la nomenclature, en
2020 il n'y aura plus qu'un seul nom qui sera `.Net`.

%Arcane supporte les deux implémentations `mono` et `coreclr`. Les
versions minimales sont `5.16` pour `mono` et `3.0` pour
coreclr. Pour les deux implémentations, il est possible de lancer du
code C# soit avec un `main` en C#, soit en mode embarqué avec un
`main` en C++.

Le mode avec le main en C# permet de lancer le code comme n'importe
quelle code C#:

- avec `mono`: `mono MyExe.dll`
- avec `dotnet`: `dotnet MyExe.dll`

Ce mode est surtout utile pour débugger, par exemple avec 'Visual
Studio Code' (TODO: faire exemple).

Le mode embarquée lance le code comme un exécutable C++ et c'est
l'appel à Arcane::ArcaneLauncher::run() qui va éventuellement charger le
runtime '.Net' et charger les assembly nécessaires.

## Bases de '.Net' {#arcanedoc_wrapping_csharp_dotnet_basics}

`.Net` est une technologie assez similaire à `java` dans son
principe. Le code source peut être écrit en plusieurs langages (C#,
F#, Visual Basic). Le code est compilé en un pseudo assembleur
(bytecode) indépendant de la plateforme. Le produit de cette
compilation s'appelle une *assembly* (équivalent aux bibliothèques
dynamiques du C++). En '.Net', l'extension est `.dll` comme les
bibliothèques dynamiques (Dynamic Loaded Library) sous Windows.

Comme java, le bytecode est lors de l'exécution convertit en code
spécifique à l'architecture de la machine cible. Le code `.Net`
nécessite la présence d'un runtime pour gérer cette partie ainsi que
pour d'autres fonctionnalités comme le Ramasse Miette (Garbage
Collector).

### Exemple en C# de main {#arcanedoc_wrapping_csharp_dotnet_example}

Par convention, les fichiers C# ont pour extension `.cs`. Le code en C#
est très similaire au code C++:

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

`.Net` utilise un outil appelé `msbuild` pour compiler et il faut
définir un projet au format XML contenant les informations
nécessaires.

\note il est aussi possible de compiler directement à la manière du
C++ mais cette méthode est moins portable car il faut spécifier
directement les assemblys nécessaires.

`msbuild` utilise un fichier projet pour définir les éléments de
compilation. Dans le principe, ce fichier projet est comme le
`Makefile` pour l'outil `make` ou le `CMakeLists.txt` pour l'outil
`CMake`. En C#, pour `msbuild`, ce fichier a par convention
l'extension `.csproj`. En général, un projet C# est créé dans un
répertoire spécifique. La commande `dotnet new` permet de créer un
répertoire avec un projet :

```sh
dotnet new console -n MyTest
```

Cela va créer un répertoire `MyTest` avec à l'intérieur un fichier
`Program.cs` et un fichier `MyTest.csproj`.

Le fichier `MyTest.csproj` sera comme suit :

```xml
 <Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>netcoreapp3.1</TargetFramework>
  </PropertyGroup>

</Project>
```

et le fichier `Program.cs` comme suit :

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

Pour compiler ce fichier, il suffit de se placer dans le répertoire du
projet et de lancer la commande `dotnet build`.

\note Par défaut, tous les fichiers ayant l'extension `.cs` présent
dans le répertoire du projet et les sous-répertoires sont
compilés. C'est pour cela qu'il est préférable de placer les projets
dans des sous-répertoires. Pour éviter ce comportement, il est possible
de mettre la valeur *false* à la propriété *EnableDefaultCompileItems*
et d'ajouter spécifiquemet les fichiers à compiler. Par exemple :

```xml
<PropertyGroup>
  <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
</PropertyGroup>
<ItemGroup>
  <Compile Include="MyFile1.cs" />
  <Compile Include="MyFile2.cs" />
</ItemGroup>
```

La commande `dotnet build` créé par défaut l'assembly dans le
répertoire `bin/${Config}/${framework}` avec *Config* ayant pour
valeur *Debug* ou *Release* et *framework* la valeur de la propriété
`msbuild` *TargetFramework*. Dans notre exemple, le répertoire sera
donc `bin/Debug/netcoreapp3.1`.

Pour exécuter le programme, il faut lancer la commande :

```sh
dotnet bin/Debug/netcoreapp3.1/MyTest.dll
```

\note il est aussi possible de lancer l'exécution directement par
`dotnet run`. Mais avant de lancer l'exécution, cette dernière
commande va vérifier s'il est nécessaire de recompiler le programme ce
qui peut prendre du temps. Si on est sûr de ne rien avoir modifié,
spécifier directement la *dll* en argument de `dotnet` est
préférable.

### Utilisation en C# d'une DLL {#arcanedoc_wrapping_csharp_dotnet_dll}

Si on ne souhaite pas avoir d'exécutable en C# mais qu'on souhaite
utiliser de code C#, alors le fonctionnement est quasi identique mais
au lieu de créer un exécutable, il faut créér une bibliothèque. Pour
cela, il suffit de créer le projet C# avec l'option `dotnet new
classlib -n MyLib`.

\note avec 'Net Core 3', il n'y a pas de différence fondamentale entre
un exécutable et une DLL. Un exécutable est juste une DLL avec un
point d'entrée bien défini (la fonction `Main`). Dans les deux cas,
l'extension aura pour nom `.dll` et un exécutable pourra être utilisé
partout où une DLL peut être utilisée.

### Ajout des références aux packages nuget Arcane {#arcanedoc_wrapping_csharp_dotnet_refnuget}

L'ajout des références aux DLL Arcane se fait via l'élément
`<PackageReference>` dans `msbuild`, par exemple:

```xml
<ItemGroup>
  <PackageReference Include="Arcane.Utils" Version="2.19.0" />
</ItemGroup>
```

Les packages `nuget` suivants sont fournies par %Arcane :

- Arcane.Utils: contient les classes utilitaires C# (tableaux et
  vues).
- Arcane.Core : contient les extensions C# des classes du coeur
  d'Arcane (les classes C++ dont les `.h` sont dans le répertoire
  'arcane').
- Arcane.Launcher : contient les classes C# pour gérer le
  lancement de code en C#. Ce package est utile si on souhaite faire
  un main en C#.
- Arcane.Services : contient les classes C# pour implémenter en
  les services de base de Arcane (Arcane::IDataWriter,
  Arcane::IDataReader, ...)
- Arcane.Hdf5: contient les classes C# correspondantes au fichier C++
  'Hdf5Utils.h'. Ce package est disponible si %Arcane a été compilé
  avec le support de `hdf5`.
- Arcane.Cea.Materials : contient les classes C# gérant les
  matériaux/milieux (uniquement CEA).

### Extension de classes C++ avec 'SWIG' {#arcanedoc_wrapping_csharp_dotnet_classesextends}

La page \ref arcanedoc_wrapping_csharp_swig décrit comment rendre accessible en C#
les classes de %Arcane et étendre d'autres classes.

### Création d'une bibliothèque .Net utilisant Arcane {#arcanedoc_wrapping_csharp_dotnet_libcreation}

Il faut tout d'abord avoir `dotnet` dans son chemin. Si %Arcane est
installé dans le répertoire `${ARCANE_PREFIX}` alors il faut lancer
les commandes suivantes :

```sh
ARCANE_PREFIX=/path/to/arcane/install
mkdir /path/to/my/project
cd /path/to/my/project
dotnet new classlib
dotnet add package Arcane.Core --source ${ARCANE_PREFIX}/nupkgs
```

Pour compiler le code et l'installer dans un répertoire, il faut exécuter la commande suivante :

```sh
dotnet publish -o out
```

Cela aura pour effet d'installer les fichiers générés dans le
répertoire `out`. Si le projet s'appelle 'toto', on aura les fichiers
suivants:

```sh
> ls  out
Arcane.Core.dll  Arcane.Utils.dll  toto.deps.json  toto.dll  toto.pdb
```

### Création d'un binaire `.Net` utilisant %Arcane {#arcanedoc_wrapping_csharp_dotnet_bincreation}

\todo A écrire.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_wrapping_csharp
</span>
<span class="next_section_button">
\ref arcanedoc_wrapping_csharp_swig
</span>
</div>
