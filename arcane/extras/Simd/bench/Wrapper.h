#ifndef BENCH_SIMD_WRAPPER_H
#define BENCH_SIMD_WRAPPER_H

#include <arcane/utils/ArrayView.h>
#include <vector>
#include "AlignedAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Cell;
class Node;
class ItemInternal;
class Item;
template<typename T> class ArrayView;
template<typename T> class ConstArrayView;
typedef ArrayView<ItemInternal*> ItemInternalArrayView;
typedef ConstArrayView<Int32> Int32ConstArrayView;
template<typename ItemType> class ItemVectorViewT;
template<typename ItemType> class ItemVectorT;
typedef ItemVectorViewT<Node> NodeVectorView;
typedef ItemVectorT<Node> NodeVector;
typedef ItemVectorViewT<Cell> CellVectorView;
typedef ItemVectorT<Cell> CellVector;
typedef ItemInternal* ItemInternalPtr;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInternal
{
 public:
  Int32 m_local_id;
  Int32 m_nb_sub_item;
  Int32* m_sub_items_lid;
  ItemInternalPtr* m_sub_internals;
};

class Item
{
 public:
  Item(ItemInternal* ii) : m_internal(ii), m_local_id(ii->m_local_id) {}
  Item(const ItemInternalPtr* internals,Integer local_id)
  : m_internal(internals[local_id]), m_local_id(local_id) {}
 protected:
  ItemInternal* m_internal;
  Int32 m_local_id;
};

class Node : public Item
{
 public:
  Node(ItemInternal* ii) : Item(ii){}
  Node(const ItemInternalPtr* internals,Integer lid) : Item(internals,lid){}
};


class Cell : public Item
{
 public:
  Cell(ItemInternal* ii) : Item(ii){}
  Cell(const ItemInternalPtr* internals,Integer lid) : Item(internals,lid){}
  inline NodeVectorView nodes();
  Int32 localId() const { return m_internal->m_local_id; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class Array
{
 public:
  Array(){}
  explicit Array(Int32 size) : m_values(size){}
  explicit Array(Int64 size) : m_values(size){}
  Array(ConstArrayView<T> view) : m_values(view.begin(),view.end()){}
 public:
  ConstArrayView<T> view() const  { return ConstArrayView<T>(m_values.size(),&m_values[0]); }
  ArrayView<T> view() { return ArrayView<T>(m_values.size(),&m_values[0]); }
  Int32 size() const { return m_values.size(); }
  const T& operator[](Int32 i) const { return m_values[i]; }
  T& operator[](Int32 i) { return m_values[i]; }
  void resize(Int32 v) { m_values.resize(v); }
 private:
  std::vector<T,AlignedAllocator<T>> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Vue sur un tableau index d'entités.
 * \see ItemVectorView
 */
class ItemInternalVectorView
{
 public:

  ItemInternalVectorView(ItemInternalArrayView items,Int32ConstArrayView local_ids)
  : m_items(items), m_local_ids(local_ids) {}

  ItemInternalVectorView(ItemInternalArrayView items,const Int32* local_ids,Integer count)
  : m_items(items), m_local_ids(count,local_ids) {}

 public:

  //! Accède au \a i-ème élément du vecteur
  inline ItemInternal* operator[](Integer index) const
  {
    return m_items[ m_local_ids[index] ];
  }

  //! Nombre d'éléments du vecteur
  inline Integer size() const { return m_local_ids.size(); }

  //! Tableau des entités
  inline ItemInternalArrayView items() const { return m_items; }

  //! Tableau des numéros locaux des entités
  inline Int32ConstArrayView localIds() const { return m_local_ids; }

 public:

 protected:

  ItemInternalArrayView m_items;
  Int32ConstArrayView m_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une liste d'entités.
 */
class ItemEnumerator
{
 protected:
  
  typedef ItemInternal* ItemInternalPtr;

 public:

  ItemEnumerator()
  : m_items(0), m_local_ids(0), m_index(0), m_count(0) {}
  ItemEnumerator(const ItemInternalPtr* items,const Int32* local_ids,Integer n)
  : m_items(items), m_local_ids(local_ids), m_index(0), m_count(n) {}
  ItemEnumerator(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids)
  : m_items(items.begin()), m_local_ids(local_ids.begin()), m_index(0), m_count(local_ids.size()) {}
  ItemEnumerator(const ItemInternalVectorView& view)
  : m_items(view.items().begin()), m_local_ids(view.localIds().begin()),
    m_index(0), m_count(view.size()) {}
  ItemEnumerator(const ItemEnumerator& rhs)
  : m_items(rhs.m_items), m_local_ids(rhs.m_local_ids),
    m_index(rhs.m_index), m_count(rhs.m_count) {}

 public:

  inline void operator++() { ++m_index; }
  inline bool operator()() { return m_index<m_count; }
  inline bool hasNext() { return m_index<m_count; }

  //! Nombre d'éléments de l'énumérateur
  inline Integer count() const { return m_count; }

  //! Indice courant de l'énumérateur
  inline Integer index() const { return m_index; }

  //! localId() de l'entité courante.
  inline Integer itemLocalId() const { return m_local_ids[m_index]; }

  //! localId() de l'entité courante.
  //inline Integer localId() const { return m_local_ids[m_index]; }

  //! Indices locaux
  inline const Int32* unguardedLocalIds() const { return m_local_ids; }

  //! Indices locaux
  inline const ItemInternalPtr* unguardedItems() const { return m_items; }

  //! Partie interne (pour usage interne uniquement)
  //inline ItemInternal* internal() const { return m_items[m_local_ids[m_index]]; }

 protected:

  const ItemInternalPtr* m_items;
  const Int32* ARCANE_RESTRICT m_local_ids;
  Integer m_index;
  Integer m_count;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une liste type d'entités
 */
template<typename ItemType>
class ItemEnumeratorT
: public ItemEnumerator
{
 public:

  ItemEnumeratorT() {}
  ItemEnumeratorT(const ItemInternalPtr* items,const Int32* local_ids,Integer n)
  : ItemEnumerator(items,local_ids,n) {}
  ItemEnumeratorT(const ItemEnumerator& rhs)
  : ItemEnumerator(rhs) {}
  ItemEnumeratorT(const ItemVectorViewT<ItemType>& rhs)
  : ItemEnumerator(rhs) {}

 public:

  ItemType operator*() const
    {
      return ItemType(m_items,m_local_ids[m_index]);
    }
  ItemType operator->() const
  {
    return ItemType(m_items[m_local_ids[m_index]]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un vecteur d'entités.
 *
 * \warning la vue n'est valide que tant que le tableau associé n'est
 * pas modifié et que la famille d'entité associée à ce tableau n'est
 * elle même pas modifiée.
 */
class ItemVectorView
{
 public:

  ItemVectorView() {}
  ItemVectorView(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids)
  : m_items(items), m_local_ids(local_ids) {}
  ItemVectorView(const ItemVectorView& view)
  : m_items(view.m_items), m_local_ids(view.m_local_ids) {}

 public:

  operator ItemInternalVectorView() const
  {
    return ItemInternalVectorView(m_items,m_local_ids);
  }

  //! Accède au \a i-ème élément du vecteur
  inline Item operator[](Integer index) const
  {
    return m_items[ m_local_ids[index] ];
  }

  //! Nombre d'éléments du vecteur
  inline Integer size() const { return m_local_ids.size(); }

  //! Tableau des entités
  inline ItemInternalArrayView items() const { return m_items; }

  //! Tableau des numéros locaux des entités
  inline Int32ConstArrayView localIds() const { return m_local_ids; }

  //! Sous-vue à partir de l'élément \a begin et contenant \a size éléments
  inline ItemVectorView subView(Integer begin,Integer size)
  {
    return ItemVectorView(m_items,m_local_ids.subView(begin,size));
  }

 public:

  inline ItemEnumerator enumerator() const;

 protected:
  
  ItemInternalArrayView m_items;
  Int32ConstArrayView m_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un tableau typé d'entités.
 */
template<typename ItemType>
class ItemVectorViewT
: public ItemVectorView
{
 public:

  ItemVectorViewT() : ItemVectorView() {}
  ItemVectorViewT(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids)
  : ItemVectorView(items,local_ids) {}
  ItemVectorViewT(const ItemVectorViewT<ItemType>& rhs)
  : ItemVectorView(rhs) {}
  ItemVectorViewT(const ItemVectorView& rhs)
  : ItemVectorView(rhs) {}
  inline ItemVectorViewT(const ItemVectorT<ItemType>& rhs);

 public:

  ItemType operator[](Integer index) const
  {
    return ItemType(m_items.begin(),m_local_ids[index]);
  }

 public:
  
  inline ItemEnumeratorT<ItemType> enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_items.begin(),m_local_ids.begin(),
                                     m_local_ids.size());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemVector
{
 public:

  typedef Item ItemType;

  ItemVector(ItemInternalArrayView items,Int32ConstArrayView local_ids)
  : m_items(items), m_local_ids(local_ids) { }
  ItemVector(){}

 public:

  //! Enumérateur
  inline ItemEnumerator enumerator() const
  {
    return ItemEnumerator(m_items.begin(),&m_local_ids[0],m_local_ids.size());
  }

  inline Int32 size() const { return m_items.size(); }

 protected:

  ItemInternalArrayView m_items;
  Array<Int32> m_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur typé d'entité.
 *
 * Pour plus d'infos, voir ItemVector.
 */
template<typename VectorItemType>
class ItemVectorT
: public ItemVector
{
 public:

  typedef VectorItemType ItemType;

 public:

  //! Constructeur vide
  ItemVectorT()
  : ItemVector() {}
  ItemVectorT(ItemInternalArrayView items,Int32ConstArrayView local_ids)
  : ItemVector(items, local_ids) { }
  ItemVectorT(const ItemVector &rhs)
  : ItemVector(rhs) { }
  virtual ~ItemVectorT() { }

 public:

  //! Operateur de cast vers ItemVectorView
  operator ItemVectorViewT<VectorItemType>() const { return ItemVectorViewT<VectorItemType>(m_items,m_local_ids); }

 public:

  //! Entité à la position \a index du vecteur
  ItemType operator[](Integer index) const
  {
    return m_items[m_local_ids[index]];
  }

  //! Vue sur le tableau entier
  ItemVectorViewT<ItemType> view()
  {
    return ItemVectorViewT<ItemType>(m_items,m_local_ids);
  }

  //! Vue sur le tableau entier
  ItemVectorViewT<ItemType> view() const
  {
    return ItemVectorViewT<ItemType>(m_items,m_local_ids);
  }

  //! Enumérateur
  ItemEnumeratorT<ItemType>	enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_items.begin(),&m_local_ids[0],
                                     m_local_ids.size());
  }

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemVectorViewT<ItemType>::
ItemVectorViewT(const ItemVectorT<ItemType>& rhs)
: ItemVectorView(rhs.view())
{  
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline NodeVectorView Cell::nodes()
{
  Integer nb_sub = m_internal->m_nb_sub_item;
  //throw std::exception();
  return NodeVectorView(ArrayView<ItemInternal*>(nb_sub,m_internal->m_sub_internals),
                        Int32ConstArrayView(nb_sub,m_internal->m_sub_items_lid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef ItemEnumeratorT<Node> NodeEnumerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
