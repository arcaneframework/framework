#ifndef BENCH_SIMD_VARIABLES_H
#define BENCH_SIMD_VARIABLES_H

#include "Wrapper.h"
#include "arcane/utils/Real3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ENUMERATE_SIMD_ITEM(iter,simd_item) \
  for( Integer iter(0); iter<simd_item.nbValid(); ++iter )

#define ENUMERATE_CELL_SIMD(name,group) \
  for( ::Arcane::SimdItemEnumerator2T< ::Arcane::Cell > name((group).enumerator()); name.hasNext(); ++name )

#define ENUMERATE_NODE(name,group) \
for( ::Arcane::ItemEnumeratorT< ::Arcane::Node > name((group).enumerator()); name.hasNext(); ++name )

#define ENUMERATE_CELL(name,group) \
for( ::Arcane::ItemEnumeratorT< ::Arcane::Cell > name((group).enumerator()); name.hasNext(); ++name )

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
class MeshVariableScalarRefT
{
 public:
  ArrayView<DataType> asArray() { return m_values.view(); }
  ConstArrayView<DataType> asArray() const { return m_values.view(); }
  DataType& operator[](ItemType item) { return m_values[item.localId()]; }
  DataType& operator[](ItemEnumeratorT<ItemType> item){ return m_values[item.itemLocalId()]; }

  void resize(Int32 nb){ m_values.resize(nb); }
 private:

  Array<DataType> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
class MeshVariableArrayRefT
{
 public:
  ArrayView<DataType> operator[](ItemType item) { return _get(item.localId()); }
  ArrayView<DataType> operator[](ItemEnumeratorT<ItemType> iter) { return _get(iter.itemLocalId()); }
 private:
  ArrayView<DataType> _get(Int32 lid)
  {
    int idx = lid * m_dim2;
    return ArrayView<DataType>(m_dim2,&m_values[idx]);
  }
 public:
  void resize(Int32 nb,Int32 dim2){ m_dim2 = dim2; m_values.resize(nb*dim2); }
 private:
  Array<DataType> m_values;
  Int32 m_dim2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef MeshVariableArrayRefT<Node,Real3> VariableNodeArrayReal3;
typedef MeshVariableArrayRefT<Cell,Real3> VariableCellArrayReal3;

typedef MeshVariableScalarRefT<Node,Real3> VariableNodeScalarReal3;
typedef MeshVariableScalarRefT<Cell,Real3> VariableCellScalarReal3;
typedef MeshVariableScalarRefT<Cell,Real> VariableCellScalarReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
