# Lancement directe d'une exécution {#arcanedoc_execution_direct_execution}

[TOC]

Il est possible d'utiliser %Arcane sans passer par les mécanismes
utilisant les modules et la boucle en temps. Cela peut être utile pour
des codes très simples ou des utilitaires mais ce mécanisme est
déconseillé pour les gros codes de calcul car il ne permet pas
d'accéder à l'ensemble des fonctionnalités de %Arcane comme par
exemple l'équilibrage de charge, les protections/reprises ou les modules.

La page \ref arcanedoc_execution_launcher explique comment fournir les paramètre
pour initialiser %Arcane. Une fois ceci fait, il faut spécifier une fonction lambda qui
sera exécutée après l'initialisation de %Arcane. Cette lambda est
donnée en paramètre à la méthode Arcane::ArcaneLauncher::run(). Cette
fonction lambda doit retourner un `int` et prendre en paramètre une
référence à `Arcane::DirectSubDomainExecutionContext`. L'exemple suivant montre
une lambda qui affiche juste `Hello World`:

```cpp
#include <arcane/launcher/ArcaneLauncher.h>
#include <iostream>

using namespace Arcane;

int main(int argc,char* argv[])
{
  ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
  // Déclare la fonction qui sera exécutée par l'appel à run()
  auto f = [=](DirectSubDomainExecutionContext& ctx) -> int
  {
    std::cout << "Hello World\n";
    return 0;
  };
  return ArcaneLauncher::run(f);
}
```


Par défaut, le nom du fichier du jeu de données est le dernier argument de la ligne de commande:

```sh
./a.out toto.arc
```

Il est possible de spécifier directement le nom du fichier du jeu de
données via l'option `CaseDatasetFileName`.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_launcher
</span>
<span class="next_section_button">
\ref arcanedoc_execution_env_variables
</span>
</div>
