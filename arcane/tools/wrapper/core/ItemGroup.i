// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%typemap(csinterfaces) Arcane::ItemVectorView "";
%typemap(csbody) Arcane::ItemVectorView %{ %}
%typemap(SWIG_DISPOSING, methodname="Dispose", methodmodifiers="private") Arcane::ItemVectorView ""
%typemap(SWIG_DISPOSE, methodname="Dispose", methodmodifiers="private") Arcane::ItemVectorView ""
%typemap(csclassmodifiers) Arcane::ItemVectorView "public struct"
%typemap(csattributes) Arcane::ItemVectorView "[StructLayout(LayoutKind.Sequential)]"
%typemap(cstype) Arcane::ItemVectorView "Arcane.ItemVectorView"
%typemap(imtype) Arcane::ItemVectorView "Arcane.ItemVectorView"
%typemap(csin) Arcane::ItemVectorView "$csinput"
%typemap(ctype, out="Arcane::ItemVectorViewPOD") Arcane::ItemVectorView %{Arcane::ItemVectorView%}
%typemap(csout) Arcane::ItemVectorView {
    ItemVectorView ret = $imcall;$excode
    return ret;
  }
%typemap(out) Arcane::ItemVectorView
%{
   Arcane::ItemVectorView result_ref = ($1);
   $result . m_local_ids.m_local_ids.m_size = result_ref.localIds().size();
   $result . m_local_ids.m_local_ids.m_ptr = result_ref.localIds().unguardedBasePointer();
   $result . m_local_ids.m_flags = result_ref.indexes().flags();
   Arcane::ItemInternalArrayView av = result_ref.items();
   $result . m_items.m_size = av.size();
   $result . m_items.m_ptr = const_cast<Arcane::ItemInternal**>(av.unguardedBasePointer());
%}
%typemap(in) Arcane::ItemVectorView %{$1 = $input; %}
%typemap(cscode) Arcane::ItemVectorView
%{
  internal ItemInternalArrayView m_items;
  internal ItemIndexArrayView m_local_ids;

  public ItemVectorView(ItemInternalArrayView items,ItemIndexArrayView indexes)
  {
    m_items = items;
    m_local_ids = indexes;
  }

  public Int32 Size { get { return m_local_ids.Length; } }

  public Item this[Int32 index] { get { return m_items[m_local_ids[index]]; } }

  [Obsolete("Use Indexes property instead")]
  public Int32ConstArrayView LocalIds { get { return m_local_ids.LocalIds; } }

  public ItemIndexArrayView Indexes { get { return m_local_ids; } }

  [Obsolete("This method is internal to Arcane. Use Indexes or operator[] instead.")]
  public ItemInternalArrayView Items { get { return m_items; } }

  public ItemEnumerator GetEnumerator()
  {
    unsafe{return new ItemEnumerator(m_items.m_ptr,m_local_ids.m_local_ids._UnguardedBasePointer(),m_local_ids.Length);}
  }

  public ItemVectorView SubViewInterval(Integer interval,Integer nb_interval)
  {
    return new ItemVectorView(m_items,m_local_ids.SubViewInterval(interval,nb_interval));
  }

  // TODO: pour compatibilité avec l'existant. A supprimer
  internal Int32ConstArrayView _LocalIds { get { return m_local_ids.LocalIds; } }
  internal ItemInternalArrayView _Items { get { return m_items; } }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%typemap(csinterfaces) Arcane::ItemEnumerator "";
%typemap(csbody) Arcane::ItemEnumerator %{ %}
%typemap(SWIG_DISPOSING, methodname="Dispose", methodmodifiers="private") Arcane::ItemEnumerator ""
%typemap(SWIG_DISPOSE, methodname="Dispose", methodmodifiers="private") Arcane::ItemEnumerator ""
%typemap(csclassmodifiers) Arcane::ItemEnumerator "public unsafe struct"
%typemap(csattributes) Arcane::ItemEnumerator "[StructLayout(LayoutKind.Sequential)]"
%typemap(cstype) Arcane::ItemEnumerator "Arcane.ItemEnumerator"
%typemap(imtype) Arcane::ItemEnumerator "Arcane.ItemEnumerator"
%typemap(csin) Arcane::ItemEnumerator "$csinput"
%typemap(ctype, out="Arcane::ItemEnumeratorPOD") Arcane::ItemEnumerator %{Arcane::ItemEnumerator%}
%typemap(csout) Arcane::ItemEnumerator {
    ItemEnumerator ret = $imcall;$excode
    return ret;
  }
%typemap(out) Arcane::ItemEnumerator
%{
  Arcane::ItemEnumerator result_ref = ($1);
  $result . m_items = result_ref.unguardedItems();
  $result . m_local_ids = result_ref.unguardedLocalIds();
  $result . m_index = result_ref.index();
  $result . m_count = result_ref.count();
%}
%typemap(in) Arcane::ItemEnumerator %{$1 = $input; %}
%typemap(cscode) Arcane::ItemEnumerator
%{
  internal ItemInternal** m_items;
  internal Int32* m_local_ids;
  internal Integer m_current;
  internal Integer m_end;

  public ItemEnumerator(ItemInternal** items,Int32* local_ids,Integer end)
  {
    m_items = items;
    m_local_ids = local_ids;
    m_current = -1;
    m_end = end;
  }

  public void Reset()
  {
    m_current = -1;
  }

  public Item Current
  {
    get{ return new Item(m_items[m_local_ids[m_current]]); }
  }

  public bool MoveNext()
  {
    ++m_current;
    return m_current<m_end;
  }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%typemap(cscode) Arcane::ItemGroup
%{
  public new ItemEnumerator GetEnumerator()
  {
    ItemEnumerator e = _enumerator();
    e.Reset();
    return e;
  }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_ITEMGROUP_SPECIALIZE(CTYPE)

%typemap(cscode) Arcane::ItemGroupT<Arcane::CTYPE>
%{
  public new IndexedItemEnumerator<CTYPE> GetEnumerator()
  {
    IndexedItemEnumerator<CTYPE> e = new IndexedItemEnumerator<CTYPE>(_enumerator());
    e.Reset();
    return e;
  }
  public IEnumerable<ItemVectorView<CTYPE>> SubViews()
  {
    ItemVectorView group_view = View();
    for(int i=0; i<100; ++i)
      yield return new ItemVectorView<CTYPE>(group_view.SubViewInterval(i,100));
  }
%}

%typemap(cstype) Arcane::ItemEnumeratorT<Arcane::CTYPE> %{Arcane.IndexedItemEnumerator<CTYPE>%}
%typemap(ctype) Arcane::ItemEnumeratorT<Arcane::CTYPE> %{Arcane::ItemEnumeratorT<Arcane::CTYPE>%}
%typemap(imtype) Arcane::ItemEnumeratorT<Arcane::CTYPE> "Arcane.ItemEnumerator"
%typemap(csin) Arcane::ItemEnumeratorT<Arcane::CTYPE> "$csinput"
%typemap(csout) Arcane::ItemEnumeratorT<Arcane::CTYPE> {
    ItemEnumerator ret = $imcall;$excode
    return new IndexedItemEnumerator<CTYPE>(ret);
  }
%typemap(out) Arcane::ItemEnumeratorT<Arcane::CTYPE> %{ $result = ($1); %}
%typemap(in) Arcane::ItemEnumeratorT<Arcane::CTYPE> %{$1 = $input; %}

namespace Arcane
{
  template<> class ItemEnumeratorT<Arcane::CTYPE > {
    ItemEnumeratorT() {}
  };
}

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SWIG_ARCANE_ITEMGROUP_SPECIALIZE(Cell)
SWIG_ARCANE_ITEMGROUP_SPECIALIZE(Face)
SWIG_ARCANE_ITEMGROUP_SPECIALIZE(Node)
SWIG_ARCANE_ITEMGROUP_SPECIALIZE(Edge)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  class ItemEnumerator
  {
    ItemEnumerator() {}
  };
  class ItemVectorView
  {
    ItemVectorView() {}
  };
  template<> class ItemEnumeratorT;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%ignore Arcane::ItemGroupT::enumerator;
%include arcane/core/ItemGroup.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef Arcane::ItemGroupT<Arcane::Cell> CellGroup;
%template(CellGroup) Arcane::ItemGroupT<Arcane::Cell>;

typedef Arcane::ItemGroupT<Arcane::Face> FaceGroup;
%template(FaceGroup) Arcane::ItemGroupT<Arcane::Face>;

typedef Arcane::ItemGroupT<Arcane::Node> NodeGroup;
%template(NodeGroup) Arcane::ItemGroupT<Arcane::Node>;

typedef Arcane::ItemGroupT<Arcane::Edge> EdgeGroup;
%template(EdgeGroup) Arcane::ItemGroupT<Arcane::Edge>;

%template(CellGroupRangeIterator_INTERNAL) Arcane::ItemEnumeratorT<Arcane::Cell>;
%template(FaceGroupRangeIterator_INTERNAL) Arcane::ItemEnumeratorT<Arcane::Face>;
%template(NodeGroupRangeIterator_INTERNAL) Arcane::ItemEnumeratorT<Arcane::Node>;
%template(EdgeGroupRangeIterator_INTERNAL) Arcane::ItemEnumeratorT<Arcane::Edge>;
