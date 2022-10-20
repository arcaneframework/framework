# Exemple n°6 {#arcanedoc_services_modules_simplecsvoutput_example6}

[TOC]

Pour ce dernier exemple, on va voir le potentiel
des déplacements par direction.

Voici le résultat :

|Ex6|0  |1  |2  |3  |4   |5    |6    |7    |8    |9    |10   |11   |12   |13   |14   |15  |16 |17 |18 |19 |Somme 
|---|---|---|---|---|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|----|---|---|---|---|------
|0  |1  |0  |0  |0  |0   |0    |0    |0    |0    |0    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |1
|1  |1  |1  |0  |0  |0   |0    |0    |0    |0    |0    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |2
|2  |1  |2  |1  |0  |0   |0    |0    |0    |0    |0    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |4
|3  |1  |3  |3  |1  |0   |0    |0    |0    |0    |0    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |8
|4  |1  |4  |6  |4  |1   |0    |0    |0    |0    |0    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |16
|5  |1  |5  |10 |10 |5   |1    |0    |0    |0    |0    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |32
|6  |1  |6  |15 |20 |15  |6    |1    |0    |0    |0    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |64
|7  |1  |7  |21 |35 |35  |21   |7    |1    |0    |0    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |128
|8  |1  |8  |28 |56 |70  |56   |28   |8    |1    |0    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |256
|9  |1  |9  |36 |84 |126 |126  |84   |36   |9    |1    |0    |0    |0    |0    |0    |0   |0  |0  |0  |0  |512
|10 |1  |10 |45 |120|210 |252  |210  |120  |45   |10   |1    |0    |0    |0    |0    |0   |0  |0  |0  |0  |1024
|11 |1  |11 |55 |165|330 |462  |462  |330  |165  |55   |11   |1    |0    |0    |0    |0   |0  |0  |0  |0  |2048
|12 |1  |12 |66 |220|495 |792  |924  |792  |495  |220  |66   |12   |1    |0    |0    |0   |0  |0  |0  |0  |4096
|13 |1  |13 |78 |286|715 |1287 |1716 |1716 |1287 |715  |286  |78   |13   |1    |0    |0   |0  |0  |0  |0  |8192
|14 |1  |14 |91 |364|1001|2002 |3003 |3432 |3003 |2002 |1001 |364  |91   |14   |1    |0   |0  |0  |0  |0  |16384
|15 |1  |15 |105|455|1365|3003 |5005 |6435 |6435 |5005 |3003 |1365 |455  |105  |15   |1   |0  |0  |0  |0  |32768
|16 |1  |16 |120|560|1820|4368 |8008 |11440|12870|11440|8008 |4368 |1820 |560  |120  |16  |1  |0  |0  |0  |65536
|17 |1  |17 |136|680|2380|6188 |12376|19448|24310|24310|19448|12376|6188 |2380 |680  |136 |17 |1  |0  |0  |131072
|18 |1  |18 |153|816|3060|8568 |18564|31824|43758|48620|43758|31824|18564|8568 |3060 |816 |153|18 |1  |0  |262144
|19 |1  |19 |171|969|3876|11628|27132|50388|75582|92378|92378|75582|50388|27132|11628|3876|969|171|19 |1  |524288



## Point d'entrée initial

Voyons le point d'entrée `start-init` :

`SimpleTableOutputExample6Module.cc`
\snippet SimpleTableOutputExample6Module.cc SimpleTableOutputExample6_init

Dans ce point d'entrée, on ajoute des lignes et des colonnes ayant comme nom
un nombre de 0 à nombre d'itérations.  
On pense aussi à mettre la valeur "1" dans la case [0,0].



## Point d'entrée loop

Voyons le point d'entrée `compute-loop` :

`SimpleTableOutputExample6Module.cc`
\snippet SimpleTableOutputExample6Module.cc SimpleTableOutputExample6_loop

Ici, on joue avec le déplacement du pointeur. Normalement, la mise à jour du pointeur
s'effectue uniquement lors d'une écriture, pas lors d'une lecture.  
Dans cet exemple, on inverse les choses.

## Point d'entrée exit

Enfin, voyons le point d'entrée `exit` :

`SimpleTableOutputExample6Module.cc`
\snippet SimpleTableOutputExample6Module.cc SimpleTableOutputExample6_exit

Même point d'entrée que précédemment.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example5
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example7
</span> -->
</div>
