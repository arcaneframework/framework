Pour utiliser les exemples, recopier ce repertoire 'samples'
chez vous et compiler chaque exemple.

Pour compiler un exemple, se placer dans le repertoire correspondant
et lancer la commande

  gmake

Pour tester un exemple, se placer dans le repertoire de
cet exemple et lancer l'executable (extension .exe) en
specifiant en argument le jeu de donnees (extension .arc).

Par exemple, pour l'exemple poisson:

  cd poisson
  gmake
  ./Poisson.exe Poisson.arc

Les sorties du cas sont dans le repertoire 'output'. Dans
ce repertoire, un repertoire 'courbes' contient les courbes
par iterations et le repertoire 'depouillement' le maillage et
les variables pour le post-traitement.

Vous pouvez ajouter les options suivantes pour chaque exemple. Les
options doivent etre ajoutees avant le jeu de donnees (qui doit
toujours etre le dernier argument).

 -arcane_opt max_iteration N  

     avec N le nombre d'iterations a effectuer

 -arcane_opt continue

     pour faire une reprise: continuer une execution precedente.


Pour lancer un cas en parallele, il faut specifier le service
de parallelisme via la variable d'environnement ARCANE_PARALLEL_SERVICE.
Les valeurs possibles sont: 'Mpi', 'Thread', 'MpiThread'.
Dans le cas ou on utilise des threads, il faut specifier leur nombre
via la variables d'environnement ARCANE_NB_THREAD.

Par exemple, pour 3 process MPI:

 export ARCANE_PARALLEL_SERVICE=Mpi
 mpiexec -n 3 ./Poisson.exe Poisson.arc

pour 4 threads:

 export ARCANE_PARALLEL_SERVICE=Thread
 export ARCANE_NB_THREAD=4
 ./Poisson.exe Poisson.arc

pour 3 process MPI et 4 threads (soit 12 sous-domaines)

 export ARCANE_PARALLEL_SERVICE=MpiThread
 export ARCANE_NB_THREAD=4
 mpiexec -n 3 ./Poisson.exe Poisson.arc
