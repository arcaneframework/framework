# Utilisation de fonctions C# pour le jeu de données {#arcanedoc_wrapping_csharp_casefunction}

[TOC]

Il est possible à partir de la version 3.11 de %Arcane de définir en
C# des fonctions du jeu données. Il existe deux types de fonctions :

- les fonctions classiques utilisées automatiquement par %Arcane
- les fonctions avancées qui doivent être appelées explicitement par
  le développeur.

L'exemple 'user_function' montre comment utiliser les fonctions
utilisateurs en C#.

## Fonctions simples

Pour définir des fonctions du jeu de données en C#, l'utilisateur doit
définir une classe contenant un ensemble de méthodes publiques dont la
signature correspond à celle d'une fonction du jeu de données. Lorsque
c'est le cas %Arcane génère une fonction du jeu de données dont le nom
est celui de la méthode correspondante.

Les signatures valides sont celles correspondantes aux méthodes
prenant un argument de type `double` ou `Int32` et retournant un type
`double` ou `Int32`.

Par exemple, avec le code suivant, il y aura 2 fonctions utilisateurs

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

## Fonctions avancées

Les fonctions avancées doivent être appelées directement par le code
C++. L'interface \arcane{IStandardFunction} permet de récupérer un
pointeur sur ces méthodes. Il existe 4 prototypes possibles :

- f(Real,Real) -> Real
- f(Real,Real3) -> Real
- f(Real,Real) -> Real3
- f(Real,Real3) -> Real3

Pour une option simple du jeu de données, il est possible de récuperer
l'instance de \arcane{IStandardFunction} via la méthode
\arcane{CaseOptionSimple::standardFunction()}.

Par exemple si l'option `node-velocity` du jeu de données doit avoir
une fonction dont la signature est `f(Real,Real3) -> Real3`, on peut
récupérer l'instance et l'utiliser comme cela:

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

L'instance `functor` n'est pas modifiée au cours du calcul. Il est
donc possible de la conserver entre les itérations.

## Compilation et utilisation

La commande `arcane_dotnet_compile` disponible dans le répertoire
`bin` d'installation de %Arcane permet de compiler un fichier C# en
une assembly (`.dll`) qui sera placée dans le répertoire courant. Le
nom de l'assembly est le nom du fichier compilé sans l'extension
`.cs`. Le fichier compilé est indépendant de la plateforme cible et il
n'est donc nécessaire de la compiler qu'une seule fois.

```{sh}
> ls .
Functions.cs
> arcane_dotnet_compile Functions.cs
> ls .
Functions.cs  Functions.dll  Functions.pdb
```

Il faut ensuite référencer dans le fichier `.arc` cet assembly dans
l'élément `<functions>` à la racine du jeu de données:

```{xml}
<functions>
  <external-assembly>
    <assembly-name>Functions.dll</assembly-name>
    <class-name>MyDotNetFunctions</class-name>
  </external-assembly>
</functions>
```

Lors de l'exécution du calcul, il faut spécifier l'argument
`-A,UsingDotNet=1` pour que l'environnement `.Net` soit utilisé et que
l'assembly `Functions.dll` soit chargée. Si le nom de l'assembly est
relatif (par exemple sous Linux il ne commence pas par `/`) ,
l'assembly doit se trouver dans le répertoire courant d'exécution.

Une fois l'assembly chargée, %Arcane va créér une instance de la
classe spécifiée dans l'élément `<class-name>`. Il est donc possible
d'avoir un constructeur dans cette classe et ce dernier sera donc
utilisé. La classe peut se trouver dans un `namespace`. Dans ce cas il
faut indiquer le nom complet dans `<class-name>` comme par exemple
`MyNamespace.MyClass`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_wrapping_csharp_swig
</span>
<span class="next_section_button">
\ref arcanedoc_wrapping_python
</span>
</div>
