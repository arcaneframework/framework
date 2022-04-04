# Exemples Arcane

## Compilation et tests des exemples

En supposant que Arcane est installé dans le répertoire
`${INSTALL_PATH}`, il est possible de recopier et de compiler les
exemples en suivant la procédure suivante:

~~~{.sh}
# Recopie les exemples dans /tmp/samples
cp -r ${INSTALL_PATH}/samples /tmp
cd /tmp/samples
cmake .
cmake --build .
~~~

La commande `ctest` permet de lancer les tests sur les exemples.

Il est aussi possible de lancer directement un exemple en se placant
dans le répertoire concerné et en lancant une des commandes
suivantes. A noter que certains exemples ne supportent certaines
combinaisons

~~~{.sh}
cd microhydro
# Lancement séquentiel
./MicroHydro MicroHydro.arc

# Lancement séquentiel avec au maxmimum 30 itérations
./MicroHydro -A,MaxIteration=30 MicroHydro.arc

# Lancement parallèle avec MPI sur 4 processus et 4 sous-domaines
# (Attention à bien utiliser le `mpiexec` qui correspond à la version
#  de MPI avec laquelle Arcane a été compilé)
mpiexec -n 4 ./MicroHydro MicroHydro.arc

# Lancement avec 1 sous-domaine et 4 threads pour les tâches:
./MicroHydro -A,T=4 MicroHydro.arc

# Lancement avec 4 sous-domaines en mémoire partagée (1 processus et 1
# thread par sous-domaine)
./MicroHydro -A,S=4 MicroHydro.arc

# Lancement avec 12 sous-domaines, dont 3 processus et pour chaque
# processus 4 sous-domaines en mémoire partagée
mpiexec -n 3 ./MicroHydro -A,S=4 MicroHydro.arc
~~~

Les sorties des cas sont dans le repertoire 'output'. Dans
ce repertoire, un repertoire 'courbes' contient les courbes
par iterations et le repertoire 'depouillement' le maillage et
les variables pour le post-traitement.
