# Transition to version 2.0 {#arcanedoc_news_cleanup_v2}

When transitioning to version 2.0, it is planned to permanently remove certain
classes that have been obsolete for several years and to slightly modify the
behavior of other classes.

The following table lists the classes that will be removed and how to replace
them.

<table>
<tr>
<td>ConstCString</td>
<td>To be replaced by the String class</td>
</tr>
<tr>
<td>CString</td>
<td>To be replaced by the String class</td>
</tr>
<tr>
<td>CStringAlloc</td>
<td>To be replaced by the String class</td>
</tr>
<tr>
<td>CStringBufT</td>
<td>To be replaced by the String class</td>
</tr>
<tr>
<td>OCStringStream</td>
<td>To be replaced by the OStringStream class</td>
</tr>
<tr>
<td>CArrayT</td>
<td>To be replaced by the UniqueArray class</td>
</tr>
<tr>
<td>BufferT</td>
<td>To be replaced by the UniqueArray class</td>
</tr>
<tr>
<td>CArrayBaseT</td>
<td>To be replaced by the ArrayView class</td>
</tr>
<tr>
<td>ConstCArrayT</td>
<td>To be replaced by the ConstArrayView class</td>
</tr>
<tr>
<td>CArray2T</td>
<td>To be replaced by the UniqueArray2 or UniqueMultiArray2 class</td>
</tr>
<tr>
<td>CArray2BaseT</td>
<td>To be replaced by the Array2View or MultiArray2View class</td>
</tr>
<tr>
<td>CArrayBuilderT</td>
<td>To be replaced by SharedArray</td>
</tr>
<tr>
<td>MutableArray</td>
<td>To be replaced by SharedArray</td>
</tr>
<tr>
<td>ConstArray</td>
<td>To be replaced by SharedArray</td>
</tr>
</table>

Version 2.0 also includes the following changes:

- The String class becomes immutable. Operators allowing modification of the
  instance, such as String::operator+=(), are removed.
- The Array and Array2 classes have been modified to prohibit copies. Indeed,
  the default behavior, which had a reference semantics, was not explicit and
  could mislead people accustomed to standard STL classes such as std::vector.
  Therefore, you must now use SharedArray or UniqueArray instead of Array if you
  wish to be able to copy the array. The SharedArray class uses reference
  semantics and the UniqueArray class uses value semantics. Both these classes
  derive from Array and can therefore be used when you need to pass a mutable
  array. For more information, the page \ref arcanedoc_core_types_array_usage
  describes the use of array-managing classes in %Arcane.
- Following the prohibition of copies for Array and Array2, methods that used an
  Array or Array2 as an argument now use an Array& or Array2&.
