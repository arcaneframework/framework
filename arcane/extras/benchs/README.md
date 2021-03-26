# Benchs

## Lancement direct des tests

L'exécutable permettant de lancer les tests est
`lib/arcane_tests_exec`. Pour lancer un test, il suffit de spécifier
cet exécutable et le jeu de données à utiliser (fichiers avec
l'extension `.arc`). Par exemple:

```{.sh}
arcane_tests_exec hydro.arc
```

Arcane utilise un mécanisme de partitionnement de maillage pour
distribuer le maillage sur plusieurs coeurs de calcul. Chaque coeur
récupère un morceau du maillage (appelé sous-domaine) et les communications entre ces
morceaux de maillage se font par échange de message. Il y a 3
implémentations pour l'échange de message. Chaque mode est géré par un
service Arcane dont le nom est affiché au début du listing. Par
exemple:

```{.txt}
*I-Internal   MessagePassing service=HybridParallelSuperMng
```

Il est donc possible de lancer un cas tests suivants 4 modes: les 3
modes d'échange de message et le mode séquentiel:

- le mode séquentiel (MpiSequentialSuperMng)
- le mode MPI pure (MpiParallelSuperMng)
- le mode mémoire partagé pure (MpiSharedMemoryParallelSuperMng)
- le mode hybride MPI + mémoire partagée (HybridParallelSuperMng). Ce
  mode nécessite pour MPI le support de MPI_THREAD_MULTIPLE (Arcane
  peut éventuellement utiliser un verrou interne si seul le mode
  MPI_THREAD_SERIALIZED est disponible)

Par défaut seul le sous-domaine 0 affiche le listing. La détection du
nombre de sous-domaines gérés par MPI se fait automatiquement. Il est
donc juste nécessaire de spécifier le nombre de sous-domaines gérés en
mémoire partagée. Cela se fait via l'option 'S' de la ligne de
commande. Il est possible de vérifier qu'on a le bon nombre de
sous-domaine dans le listing en regardant la ligne qui contient
'Subdomain number'. Par exemple:

```{.sh}
# Lance le calcul avec 5 sous-domaines gérés en mémoire partagée
arcane_tests_exec -A,S=5 hydro.arc
...
*I-Internal   MessagePassing service=MpiSharedMemoryParallelSuperMng
...
*I-Init       Subdomain number is 0/5
```

```{.sh}
# Lance le calcul avec 3 sous-domaines gérés via MPI
mpiexec -n 3 arcane_tests_exec hydro.arc
...
*I-Internal   MessagePassing service=MpiParallelSuperMng
...
*I-Init       Subdomain number is 0/3
```

```{.sh}
# Lance le calcul avec 4 sous-domaines gérés via MPI et pour
# chaque processus MPI 5 sous-domaines gérés par de la mémoire partagée.
# On aura donc en tout 4*5 = 20 sous-domaines

mpiexec -n 4 arcane_tests_exec -A,S=5 hydro.arc
...
*I-Internal   MessagePassing service=HybridParallelSuperMng
...
*I-Init       Subdomain number is 0/20
```

## Benchs disponibles

### MicroHydro

Ce bench contient une hydro très simplifiée sur un tube à choc. En
parallèle, chaque sous-domaine contient le même morceau de tube qui
est placé au dessus du sous-domaine précédent. Chaque sous-domaine
n'est donc relié qu'à deux sous-domaines au maximun. Le nombre de
mailles de chaque sous-domaine est spécifié dans le jeu de données
`hydro.arc`:

```{.xml}
<meshgenerator><sod><x>100</x><y>15</y><z>15</z></sod></meshgenerator>
```

L'exemple précédent contient 100x15x15 soit 25000 mailles. En
parallèle, le nombre total de mailles est donc 25000 fois le nombre de
sous-domaines. Le cas test est dimensionné pour faire 100
itérations. Ce nombre dépend du nombre de mailles en X. Il ne faut
donc pas le changer. Il est par contre possible de changer le nombre
de mailles en Y ou Z si on souhaite réduire ou grossir le cas.

Ce test permet donc de vérifier l'extensibilité faible. La valeur
importante est le temps de calcul de la boucle. Le temps de
l'initialisation n'est pas significatif. La ligne suivante vers la
fin du listing indique le temps de calcul de la boucle:

```
*I-Internal   TotalReel = 19.7302148342133 secondes (init: 8.42602610588074  loop: 1.3041887283325 )
```

La valeur importante est celle qui suit `loop:`. Si l'extensibilité
est bonne, cette valeur ne doit pas changer en fonction du nombre de
sous-domaines.

### Particles

Ce bench comporte deux modes:

- un mode synchrone
- un mode asynchrone

Ce bench n'a pas vocation à tester l'extensibilité des performances
mais plutôt la robustesse de l'implémentation MPI, notamment en mode
asynchrone où le nombre de messages peut être important.

Il est possible comme pour microhydro de changer le nombre de mailles.

Il est aussi possible de changer le nombre de particules par
mailles. Augmenter cette valeur permet d'avoir des messages plus
gros. La ligne suivante du jeu de données permet de changer cette
valeur:

```{.xml}
<nb-particule-par-maille>2</nb-particule-par-maille>
```

