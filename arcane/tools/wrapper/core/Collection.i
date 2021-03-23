// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Gestion des collections.
 *
 * Les collections comprennent les classes suivantes:
 *  - Collection
 *  - Enumerator
 *  - List
 *
 * Pour être conforme au C#, les collections et énumérateurs associés
 * implémentent ICollection<T> et IEnumerator<T>.
 //FIXME peut etre faut-il aussi implementer IList<T> pour les listes.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  template<typename T> class EnumeratorT;

  template<typename T> class Collection
  {
   public:
    void clear();
    Integer count() const;
    bool empty() const;
    bool remove(const T& value);
    void add(const T& value);
    bool contains(const T& value) const;
    EnumeratorT<T> enumerator();
  };

  template<typename T> class EnumeratorT
  {
   public:
    void reset();
    bool moveNext();
    const T& current();
  };

  template<typename T> class List : public Collection<T>
  {
   public:
    List();
    List(const ConstArrayView<T>& from);
    List(const ArrayView<T>& from);
   public:
    void resize(Integer new_size);
    List<T> clone() const;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Macro pour définir un énumérateur implémentant IEnumerator:
// - ENUMERATOR_TYPE: classe C++ de l'énumérateur
// - CSTYPE: type C# de l'objet énuméré
%define SWIG_ARCANE_COLLECTION_ENUMERATOR(ENUMERATOR_TYPE,CSTYPE)

%typemap(csinterfaces) ENUMERATOR_TYPE "System.Collections.Generic.IEnumerator<CSTYPE>";
%typemap(cscode) ENUMERATOR_TYPE
%{
  public CSTYPE Current { get { return _current(); } }
  CSTYPE IEnumerator<CSTYPE>.Current { get{ return _current(); } }
  object IEnumerator.Current { get { return _current(); } }
%}

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_COLLECTION(CTYPE,CSTYPE,COLLECTION_NAME)
SWIG_ARCANE_COLLECTION_ENUMERATOR(Arcane::EnumeratorT<CTYPE >,CSTYPE)

%typemap(csinterfaces) Arcane::Collection<CTYPE > "System.Collections.Generic.ICollection<CSTYPE>";
%typemap(cscode) Arcane::Collection<CTYPE >
%{
  public int Count { get { return (int)_count(); } }
  public bool IsEmpty { get { return _empty(); } }
  public bool IsReadOnly { get { return false; } }
  public COLLECTION_NAME##CollectionEnumerator GetEnumerator() { return _enumerator(); }
  IEnumerator<CSTYPE> IEnumerable<CSTYPE>.GetEnumerator() { return _enumerator(); }
  IEnumerator IEnumerable.GetEnumerator() { return _enumerator(); }
  
  public void CopyTo(CSTYPE[] array,int index)
  {
    foreach(CSTYPE s in this){
      array[index] = s;
      ++index;
    }
  }
%}

%template(COLLECTION_NAME##Collection) Arcane::Collection<CTYPE>;
%template(COLLECTION_NAME##CollectionEnumerator) Arcane::EnumeratorT<CTYPE>;
%template(COLLECTION_NAME##List) Arcane::List<CTYPE>;

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SWIG_ARCANE_COLLECTION(ISession*,ISession,Session);
SWIG_ARCANE_COLLECTION(String,string,String);
SWIG_ARCANE_COLLECTION(ISubDomain*,ISubDomain,SubDomain);
SWIG_ARCANE_COLLECTION(ItemGroup,ItemGroup,ItemGroup);
SWIG_ARCANE_COLLECTION(IItemFamily*,IItemFamily,ItemFamily);
SWIG_ARCANE_COLLECTION(IMesh*,IMesh,Mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Specialisation pour la classe VariableCollection
// A la différence des autres collections, celle-ci est en lecture seule
// et il n'est donc pas possible de modifier ses éléments

SWIG_ARCANE_COLLECTION_ENUMERATOR(Arcane::VariableCollectionEnumerator,Arcane.IVariable)

// Normalement il faudrait utiliser IReadOnlyCollection mais cela n'est pas
// disponible avec les anciennes versions de .NET
%typemap(csinterfaces) Arcane::VariableCollection "System.Collections.Generic.IReadOnlyCollection<Arcane.IVariable>";
%typemap(cscode) Arcane::VariableCollection
%{
  public int Count { get { return (int)_count(); } }
  public bool IsEmpty { get { return _empty(); } }
  public bool Remove(IVariable value) { throw new NotSupportedException(); }
  public void Add(IVariable value) { throw new NotSupportedException(); }
  public bool IsReadOnly { get { return true; } }
  public VariableCollectionEnumerator GetEnumerator() { return _enumerator(); }
  IEnumerator<IVariable> IEnumerable<IVariable>.GetEnumerator() { return _enumerator(); }
  IEnumerator IEnumerable.GetEnumerator() { return _enumerator(); }

  public void CopyTo(IVariable[] array,int index)
  {
    foreach(IVariable s in this){
      array[index] = s;
      ++index;
    }
  }
%}

%include arcane/VariableCollection.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
