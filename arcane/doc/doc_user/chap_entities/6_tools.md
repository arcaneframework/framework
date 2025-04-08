# Outils externes de gestion de maillage {#arcanedoc_entities_tools}

[TOC]

%Arcane propose deux outils pour convertir ou partitionner un maillage.

## Outil de partitionnement {#arcanedoc_entities_tools_arcane_partition_mesh}

L'outil `arcane_partition_mesh` permet de partitionner un maillage
existant et de générer un fichier de maillage par partie. Il s'agit
d'un outil parallèle qui utilise par défaut `ParMetis` pour réaliser
le partitionnement. Cet outil est installé dans le répertoire `bin`
de l'installation de %Arcane.

L'usage est le suivant:

```
${ARCANE_INSTALL_ROOT}/bin/arcane_partition_mesh
  -n nb_proc
  -p nb_part
  --writer writer_service_name
  [--manifold-]
  [-Wp,arg1,arg2,...]
  mesh_file_name
```

Les valeurs possibles pour `writer_service_name` sont des services qui
implémentent \arcane{IMeshWriter}. Par exemple:

- `MshMeshWriter` (le défaut)
- `LimaMeshWriter`
- `VtkLegacyMeshWriter` : uniquement pour visualiser le résultat. Les
  fichiers générés ne pourront pas être utilisés en lecture du code.

Le fichier de maillage d'entrée (`mesh_file_name`) peut être au format MSH (extension
`.msh`), au format VTK history (extension `.vtk`) ou un format
supporté par `Lima` (extension `.mli2`, `.mli`, `.unf`).
Lorsque le maillage d'entrée est au format MSH, il n'est pas possible
de savoir à priori si le maillage est non-manifold. Par défaut %Arcane
suppose qu'il s'agit d'un maillage manifold. Si ce n'est pas le cas il
faut spécifier l'option `--manifold-` dans la ligne de commande.

Le partitionneur se lancera sur `nb_proc` processus (via MPI) et va
générer `nb_part` partitions. Le nombre de partitions doit être un
multiple du nombre de processus utilisés. En sortie, il y a aura un
fichier par partie. Les fichiers de sortie seront de la
forme `CPU00000`, `CPU00001`, ... avec l'extension correspondante au
format du maillage (par exemple `.msh` pour le format MSH).

L'option `-Wp` permet de fournir des arguments pour le lanceur
parallèle (`mpiexec` ou `srun` en général). Par exemple, la valeur
`-Wp,-c,4` permet d'ajouter `-c 4` au lanceur parallèle.

### Gestion du format MSH

\warning La lecture et la sortie au format MSH est expérimental depuis
la version 3.16.0 de %Arcane et comporte actuellement un certain nombre de
limitations.

Si le format d'entrée et de sortie est MSH, chaque fichier contiendra les mêmes
`$Entities` que le format d'origine ainsi que les informations de
périodicité (`$Periodics`) pour la partie concernée.
Il y a actuellement les limitations suivantes:

- Seules les entités d'ordre 1 sont gérées
- Les informations de périodicité autre que les couples de noeuds
  (maitre,esclave) sont cohérentes. Les informations affines ou sur
  les entités ne sont pas significatives. Pour les couples
  (maitre,esclave), le couple sera présent dans le fichier si un des
  deux noeuds est présent dans la partie.
- les informations sur les boîtes englobantes pour les `$Entities` ne
  sont pas gérées
- les blocs d'origine du format MSH ne sont pas forcément sauvegardés
  et les numéros des `$Entities` d'origine ne sont pas conservés.

### Utilisation des fichiers pré-découpés

TODO

### Exemples d'utilisation

L'exemple suivant utilise 2 processeurs pour découper le maillage `onesphere.msh` en 4
parties. Le fichier de sortie sera au format VTK.

```
./bin/arcane_partition_mesh -n 2 -p 4 --writer VtkLegacyMeshWriter onesphere.msh
ls
CPU00000.vtk
CPU00001.vtk
CPU00002.vtk
CPU00003.vtk
```

L'exemple suivant utilise 4 processeurs pour découper le maillage
`mesh_with_loose_items.msh` en 12 parties. Le fichier de sortie sera
au format MSH. Le maillage d'entrée étant non-manifold, on utilise
l'option `--manifold-` pour le spécifier.

```
./bin/arcane_partition_mesh -n 4 -p 12 --writer MshMeshWriter --manifold- mesh_with_loose_items.msh
ls
CPU00000.msh
CPU00001.msh
...
CPU00010.msh
CPU00011.msh
CPU00012.msh
```

## Outils de conversion de maillage {#arcanedoc_entities_tools_arcane_convert_mesh}

L'outil `arcane_convert_mesh` permet de convertir un fichier de
maillage d'un format vers un autre.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_geometric
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_execution_direct_execution
</span> -->
</div>
