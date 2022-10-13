# Passage à la version 2.0 {#arcanedoc_news_cleanup_v2}

Lors du passage à la version 2.0, il est prévu de supprimer
définitivement certaines classes qui sont obsolètes depuis plusieurs
années et de modifier quelque peu le comportement d'autres classes.

Le tableau suivant liste les classes qui seront supprimées et
comment les remplacer.

<table>
<tr>
<td>ConstCString</td>
<td>A remplacer par la classe String</td>
</tr>
<tr>
<td>CString</td>
<td>A remplacer par la classe String</td>
</tr>
<tr>
<td>CStringAlloc</td>
<td>A remplacer par la classe String</td>
</tr>
<tr>
<td>CStringBufT</td>
<td>A remplacer par la classe String</td>
</tr>
<tr>
<td>OCStringStream</td>
<td>A remplacer par la classe OStringStream</td>
</tr>
<tr>
<td>CArrayT</td>
<td>A remplacer par la classe UniqueArray</td>
</tr>
<tr>
<td>BufferT</td>
<td>A remplacer par la classe UniqueArray</td>
</tr>
<tr>
<td>CArrayBaseT</td>
<td>A remplacer par la classe ArrayView</td>
</tr>
<tr>
<td>ConstCArrayT</td>
<td>A remplacer par la classe ConstArrayView</td>
</tr>
<tr>
<td>CArray2T</td>
<td>A remplacer par la classe UniqueArray2 ou UniqueMultiArray2</td>
</tr>
<tr>
<td>CArray2BaseT</td>
<td>A remplacer par la classe Array2View ou MultiArray2View</td>
</tr>
<tr>
<td>CArrayBuilderT</td>
<td>A remplacer par SharedArray</td>
</tr>
<tr>
<td>MutableArray</td>
<td>A remplacer par SharedArray</td>
</tr>
<tr>
<td>ConstArray</td>
<td>A remplacer par SharedArray</td>
</tr>
</table>

La version 2.0 comporte aussi les modifications suivantes:
- la classe String devient non modifiable. Les opérateurs
permettant de modifier l'instance, comme String::operator+=() sont supprimés.
- Les classes Array et Array2 sont modifiées pour interdire les
recopies. En effet, le comportement par défaut qui avait une
sémantique par référence n'était pas explicite et pouvait induire en
erreur les gens habitués aux classes standards de la STL telles que
std::vector. Il faut donc maintenant utiliser SharedArray ou
UniqueArray à la place de Array si on souhaite pouvoir copier le
tableau. La classe SharedArray utilise la sémantique par référence
et la classe UniqueArray la sémantique par valeur. Ces deux classes
dérivent de Array et peuvent donc être utilisées lorsqu'il faut
passer un tableau modifiable. Pour plus d'informations, la page \ref arcanedoc_core_types_array_usage décrit
l'utilisation des classes gérant les tableaux dans %Arcane.
- Suite à l'interdiction des recopies de Array et Array2, les
méthodes qui utilisaient un Array ou Array2 en argument utilisent
maintenant un Array& ou Array2&.
