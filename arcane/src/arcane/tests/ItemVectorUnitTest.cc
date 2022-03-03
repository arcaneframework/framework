// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVectorUnitTest.cc                                       (C) 2000-2010 */
/*                                                                           */
/* Service de test des tableaux.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/FactoryService.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/IItemFamily.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/ItemVector.h"

#include "arcane/ItemPrinter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

class ItemVectorPrinter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe utilitaire pour imprimer les infos sur une entité.
 */
class ItemPrinter2
{
 private:
  static const Int32 ONLY_UID = 1 << 1;
  static const Int32 CONNECTIVITY = 1 << 2;
 public:
  ItemPrinter2(ItemInternal* item,eItemKind ik)
  : m_item(item), m_item_kind(ik), m_has_item_kind(true)
  { _init(); }
  ItemPrinter2(ItemInternal* item)
  : m_item(item), m_item_kind(IK_Unknown), m_has_item_kind(false)
  { _init(); }
  ItemPrinter2(const Item& item)
  : m_item(item.internal()), m_item_kind(IK_Unknown), m_has_item_kind(false)
  { _init(); }
  ItemPrinter2(const Item& item,eItemKind ik)
  : m_item(item.internal()), m_item_kind(ik), m_has_item_kind(true)
  { _init(); }
  ItemPrinter2()
  : m_item(0), m_item_kind(IK_Unknown), m_has_item_kind(false)
  { _init(); }

 public:

  void print(std::ostream& o) const;

  ItemPrinter2& assign(Item item)
  {
    m_item = item.internal();
    return (*this);
  }

  ItemPrinter2& printOnlyUid()
  {
    m_flags = 0;
    m_flags |= ONLY_UID;
    return (*this);
  }

  ItemPrinter2& setIndentLevel(int indent_level)
  {
    m_indent_level = indent_level;
    return (*this);
  }

  ItemPrinter2& printConnectivity()
  {
    m_flags = 0;
    m_flags |= CONNECTIVITY;
    return (*this);
  }

 private:

  void _print(std::ostream& o,ItemVectorPrinter& ivp,ItemVectorView view,const String& name) const;

  void _printIndent(std::ostream& o) const
  {
    for( Integer z=0; z<m_indent_level*4; ++z ){
      o << ' ';
    }
  }
  void _init()
  {
    m_flags = 0;
    m_indent_level = 0;
  }

 private:

  ItemInternal* m_item;
  eItemKind m_item_kind;
  bool m_has_item_kind;
  Int32 m_flags;
  Integer m_indent_level;

};

inline std::ostream&
operator<<(std::ostream& o,const ItemPrinter2& ip)
{
  ip.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemVectorPrinter
{
 public:
  ItemVectorPrinter() : m_indent_level(0) {}
  ItemVectorPrinter(ItemVectorView iv) : m_item_vector(iv),m_indent_level(0){}
 public:
  void print(std::ostream& o) const
  {
    Integer n = m_item_vector.size();
    o << "(";
    char delim =  (m_indent_level>=0) ? '\n' : ' ';
    if (!m_name.null())
      o << m_name;
    else 
      o << "ItemVector";
    o << " size=" << n;
    o << delim;

    //if (m_indent_level>=0)
    //  m_item_printer.setIndentLevel(m_indent_level+2);
    for( Integer i=0; i<n; ++i ){
      _printIndent(o);
      m_item_printer.assign(m_item_vector[i]);
      o << "I=" << i << ' ' << m_item_printer;
      o << delim;
    }
    o << ")";
  }

  ItemVectorPrinter& assign(ItemVectorView item_vector)
  {
    m_item_vector = item_vector;
    return (*this);
  }

  ItemVectorPrinter& setName(const String& name)
  {
    m_name = name;
    return (*this);
  }

  ItemVectorPrinter& setItemPrinter(const ItemPrinter2& item_printer)
  {
    m_item_printer = item_printer;
    return (*this);
  }

  ItemVectorPrinter& setIndentLevel(int indent_level)
  {
    m_indent_level = indent_level;
    return (*this);
  }

 private:
  void _printIndent(std::ostream& o) const
  {
    for( Integer z=0; z<m_indent_level*4; ++z ){
      o << ' ';
    }
  }
 private:
  String m_name;
  ItemVectorView m_item_vector;
  mutable ItemPrinter2 m_item_printer;
  Integer m_indent_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o,const ItemVectorPrinter& ip)
{
  ip.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPrinter2::
print(std::ostream& o) const
{
  if (!m_item || m_item->localId()==NULL_ITEM_LOCAL_ID){
    o << "(null_item)";
    return;
  }
      
  if (m_flags & ONLY_UID){
    o << m_item->uniqueId();
    return;
  }

  o << "{id=" << m_item->uniqueId()
    << ':' << m_item->localId()
    << ",o=" << m_item->owner();
  if (m_has_item_kind){
    o << ",k=" << itemKindName(m_item_kind);
  }
  
  if (m_flags & CONNECTIVITY){
    eItemKind ik = m_item->kind();
    ItemVectorPrinter ivp;
    ivp.setIndentLevel(-1);
    std::cout << "KIND=" << ik << '\n';
    o << '\n';
    if (ik!=IK_Node)
      if (m_item->nbNode()!=0)
        _print(o,ivp,m_item->internalNodes(),"Nodes");
    if (m_item->nbEdge()!=0)
      _print(o,ivp,m_item->internalEdges(),"Edges");
    if (m_item->nbFace()!=0)
      _print(o,ivp,m_item->internalFaces(),"Faces");
    if (ik!=IK_Cell)
      if (m_item->nbCell()!=0)
        _print(o,ivp,m_item->internalCells(),"Cells");
    if (ik==IK_Face){
      o << "BackCell " << ItemPrinter2(m_item->backCell()) << ' ';
      o << "FrontCell " << ItemPrinter2(m_item->frontCell()) << ' ';
      o << '\n';
    }
    _printIndent(o);
  }

  o << "}";
}

void ItemPrinter2::
_print(std::ostream& o,ItemVectorPrinter& ivp,ItemVectorView view,const String& name) const
{
  if (view.size()>0){
    _printIndent(o);
    o << ivp.assign(view).setName(name) << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test des ItemVector
 */
class ItemVectorUnitTest
: public BasicUnitTest
{
 public:

  ItemVectorUnitTest(const ServiceBuildInfo& cb);
  ~ItemVectorUnitTest();

 public:

  virtual void initializeTest() {}
  virtual void executeTest();

 private:

  template<typename ItemVectorType>
  void _executeTest(IItemFamily* family);
  void _printArray(Int32ConstArrayView view);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(ItemVectorUnitTest,IUnitTest,ItemVectorUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorUnitTest::
ItemVectorUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorUnitTest::
~ItemVectorUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemVectorUnitTest::
_printArray(Int32ConstArrayView view)
{
  info() << "View n=" << view.size();
  for( Integer i=0; i<view.size(); ++i ){
    info() << "I=" << i << ' ' << view[i];
  }
}

void ItemVectorUnitTest::
executeTest()
{
  Int32UniqueArray v0(53);
  for( Integer i=0; i<v0.size(); ++i )
    v0[i] = i+1;
  Int32ArrayView v = v0.view();
  Int32ArrayView v1 = v.subView(0,5);
  Int32ArrayView v2 = v.subView(7,15);
  Int32ArrayView v3 = v.subView(15,45);
  Int32ArrayView v4 = v.subView(0,53);
  _printArray(v);
  _printArray(v1);
  _printArray(v2);
  _printArray(v3);
  _printArray(v4);
  Cell null_cell;
  info() << "UID=" << null_cell.uniqueId();

  _executeTest<CellVector>(mesh()->cellFamily());
  _executeTest<FaceVector>(mesh()->faceFamily());
  _executeTest<EdgeVector>(mesh()->edgeFamily());
  _executeTest<NodeVector>(mesh()->nodeFamily());

  _executeTest<ItemVector>(mesh()->cellFamily());

//mesh()->utilities()->writeToFile("toto.unf","Lima");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemVectorType>
void ItemVectorUnitTest::
_executeTest(IItemFamily* item_family)
{
  typedef typename ItemVectorType::ItemType ItemType;

  ItemVectorType vec(item_family);
  ENUMERATE_GENERIC(ItemType,iitem,item_family->allItems()){
    Int32 lid = iitem.itemLocalId();
    if ((lid%2)!=0){
      vec.add(lid);
      vec.addItem(*iitem);
    }
  }
  ItemVectorType vec2;

  vec2 = vec.clone();

  ItemVectorPrinter iv_printer;
  ItemPrinter2 item_printer;
  item_printer.printConnectivity().setIndentLevel(2);
  iv_printer.setItemPrinter(item_printer);
  {
    Integer sum_lid = 0;
    ENUMERATE_GENERIC(ItemType,iitem,vec.view()){
      sum_lid += iitem.itemLocalId();
    }
    info() << "** SUM_ID1 sum_lid=" << sum_lid;
    info() << iv_printer.assign(vec.view().subView(0,10)).setName(item_family->name());
  }

  {
    Integer sum_lid = 0;
    for( Integer i=0, n=vec.size(); i<n; ++i ){
      ItemType item = vec[i];
      sum_lid += item.localId();
    }
    info() << "** SUM_ID2 sum_lid="<< sum_lid;
  }
  info() << "** BEFORE REMOVE size=" << vec2.size();
  info() << iv_printer.assign(vec2.view().subView(0,30)).setName(item_family->name());
  for( Integer i=0, n=vec2.size(); i<n; ++i ){
    if (i>=vec2.size())
      break;
    if ((i%3)==0)
      vec2.removeAt(i);
  }
  info() << "** AFTER REMOVE size=" << vec2.size();
  info() << iv_printer.assign(vec2.view().subView(0,10)).setName(item_family->name());

  {
    // Teste les itérateurs sur les ItemVectorView
    Integer index = 0;
    ItemVectorView view = vec2.view();
    for( Item x : view ){
      info() << "X=" << ItemPrinter(x);
      Item ref_x = view[index];
      if (x!=ref_x)
        ARCANE_FATAL("Bad item x={0} ref={1} index={2}",ItemPrinter(x),ItemPrinter(ref_x),index);
      ++index;
    }
    ItemInternal* wanted_item_ptr = item_family->findOneItem(5);
    if (!wanted_item_ptr)
      wanted_item_ptr = ItemInternal::nullItem();
    Item wanted_item(wanted_item_ptr);

    bool is_found = std::binary_search(view.begin(),view.end(),wanted_item);
    info() << "BinarySearch found=" << is_found;
    auto iter = std::find(view.begin(),view.end(),wanted_item);
    if (iter!=view.end()){
      Item x = *iter;
      if (x.uniqueId()!=5)
        ARCANE_FATAL("Bad found item");
      info() << "FOUND! item=" << ItemPrinter(*iter);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
